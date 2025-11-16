// Function: sub_30F8D90
// Address: 0x30f8d90
//
__int64 __fastcall sub_30F8D90(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  __int64 v5; // rdx
  unsigned int v7; // ecx
  __int64 *v8; // rbx
  __int64 v9; // rdi
  _BYTE *v10; // r15
  unsigned int v11; // r14d
  char v13; // r14
  _BYTE *v14; // rax
  __int64 v15; // rbx
  __int64 v16; // rdi
  int v17; // r9d
  __int64 v18[7]; // [rsp+8h] [rbp-38h] BYREF

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a2 - 32);
  v5 = *(_QWORD *)(a1 + 8);
  if ( (_DWORD)v3 )
  {
    v7 = (v3 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v8 = (__int64 *)(v5 + 32LL * v7);
    v9 = *v8;
    if ( v4 == *v8 )
    {
LABEL_3:
      if ( v8 != (__int64 *)(v5 + 32 * v3) )
      {
        v10 = (_BYTE *)v8[1];
        if ( *v10 == 3 && !sub_B2FC80(v8[1]) && !(unsigned __int8)sub_B2F6B0((__int64)v10) )
        {
          v13 = v10[80];
          if ( (v13 & 2) == 0 )
          {
            v11 = v13 & 1;
            if ( v11 )
            {
              v14 = (_BYTE *)sub_B43CC0(a2);
              v15 = sub_9714E0(*((_QWORD *)v10 - 4), *(_QWORD *)(a2 + 8), (__int64)(v8 + 2), v14);
              if ( v15 )
              {
                v16 = *(_QWORD *)(a1 + 40);
                v18[0] = a2;
                *sub_FAA780(v16, v18) = v15;
                return v11;
              }
            }
          }
        }
      }
    }
    else
    {
      v17 = 1;
      while ( v9 != -4096 )
      {
        v7 = (v3 - 1) & (v17 + v7);
        v8 = (__int64 *)(v5 + 32LL * v7);
        v9 = *v8;
        if ( v4 == *v8 )
          goto LABEL_3;
        ++v17;
      }
    }
  }
  return 0;
}
