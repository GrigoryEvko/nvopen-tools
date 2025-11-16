// Function: sub_18E6910
// Address: 0x18e6910
//
void __fastcall sub_18E6910(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v7; // r12
  __int64 v8; // r15
  unsigned __int64 *v9; // rsi
  __int64 v10; // rbx
  __int64 v11; // rcx
  int v12; // eax
  char **v13; // r15
  __int64 v14; // rdx
  __int64 v15; // rax
  unsigned __int64 v16; // rbx
  __int64 v17; // r14
  char *v18; // rdi
  __int64 v20; // [rsp+18h] [rbp-D8h]
  char *v21[2]; // [rsp+20h] [rbp-D0h] BYREF
  _BYTE v22[128]; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v23; // [rsp+B0h] [rbp-40h]
  int v24; // [rsp+B8h] [rbp-38h]

  if ( a1 != a2 && a2 != a1 + 160 )
  {
    v7 = a1 + 320;
    do
    {
      v8 = *(_QWORD *)(v7 - 16);
      v9 = *(unsigned __int64 **)(a1 + 144);
      v20 = v7;
      v10 = v7 - 160;
      if ( *(_QWORD *)v8 == *v9 )
      {
        v9 += 3;
        if ( (int)sub_16A9900(v8 + 24, v9) < 0 )
        {
LABEL_6:
          v21[0] = v22;
          v21[1] = (char *)0x800000000LL;
          if ( *(_DWORD *)(v7 - 152) )
          {
            sub_18E63F0((__int64)v21, (char **)(v7 - 160), a3, v11, a5, a6);
            v8 = *(_QWORD *)(v7 - 16);
          }
          v12 = *(_DWORD *)(v7 - 8);
          v23 = v8;
          v13 = (char **)(v7 - 160);
          v14 = 0xCCCCCCCCCCCCCCCDLL;
          v24 = v12;
          v15 = v10 - a1;
          v16 = 0xCCCCCCCCCCCCCCCDLL * ((v10 - a1) >> 5);
          if ( v15 > 0 )
          {
            do
            {
              v17 = (__int64)v13;
              v13 -= 20;
              sub_18E63F0(v17, v13, v14, v11, a5, a6);
              *(_QWORD *)(v17 + 144) = *(_QWORD *)(v17 - 16);
              v11 = *(unsigned int *)(v17 - 8);
              *(_DWORD *)(v17 + 152) = v11;
              --v16;
            }
            while ( v16 );
          }
          sub_18E63F0(a1, v21, v14, v11, a5, a6);
          v18 = v21[0];
          *(_QWORD *)(a1 + 144) = v23;
          *(_DWORD *)(a1 + 152) = v24;
          if ( v18 != v22 )
            _libc_free((unsigned __int64)v18);
          goto LABEL_12;
        }
      }
      else
      {
        v11 = *(_DWORD *)(*(_QWORD *)v8 + 8LL) >> 8;
        if ( (unsigned int)v11 < *(_DWORD *)(*v9 + 8) >> 8 )
          goto LABEL_6;
      }
      sub_18E67F0(v7 - 160, (__int64)v9, a3, v11, a5, a6);
LABEL_12:
      a3 = v7;
      v7 += 160;
    }
    while ( a2 != v20 );
  }
}
