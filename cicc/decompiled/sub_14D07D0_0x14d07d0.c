// Function: sub_14D07D0
// Address: 0x14d07d0
//
void __fastcall sub_14D07D0(__int64 a1, __int64 a2, __int64 a3)
{
  bool v4; // zf
  __int64 v5; // r12
  __int64 v6; // rbx
  char v7; // dl
  __int64 v8; // r15
  __int64 *v9; // rax
  __int64 *v10; // rsi
  unsigned int v11; // edi
  __int64 *v12; // rcx
  unsigned __int64 v13[2]; // [rsp+10h] [rbp-1F0h] BYREF
  _BYTE v14[128]; // [rsp+20h] [rbp-1E0h] BYREF
  __int64 v15; // [rsp+A0h] [rbp-160h] BYREF
  _BYTE *v16; // [rsp+A8h] [rbp-158h]
  _BYTE *v17; // [rsp+B0h] [rbp-150h]
  __int64 v18; // [rsp+B8h] [rbp-148h]
  int v19; // [rsp+C0h] [rbp-140h]
  _BYTE v20[312]; // [rsp+C8h] [rbp-138h] BYREF

  v16 = v20;
  v4 = *(_BYTE *)(a2 + 184) == 0;
  v17 = v20;
  v13[0] = (unsigned __int64)v14;
  v15 = 0;
  v18 = 32;
  v19 = 0;
  v13[1] = 0x1000000000LL;
  if ( v4 )
    sub_14CDF70(a2);
  v5 = *(_QWORD *)(a2 + 8);
  v6 = v5 + 32LL * *(unsigned int *)(a2 + 16);
  if ( v6 != v5 )
  {
    while ( 1 )
    {
      v8 = *(_QWORD *)(v5 + 16);
      if ( !v8 )
        goto LABEL_6;
      v9 = *(__int64 **)(a3 + 8);
      if ( *(__int64 **)(a3 + 16) == v9 )
      {
        v10 = &v9[*(unsigned int *)(a3 + 28)];
        v11 = *(_DWORD *)(a3 + 28);
        if ( v9 != v10 )
        {
          v12 = 0;
          while ( v8 != *v9 )
          {
            if ( *v9 == -2 )
              v12 = v9;
            if ( v10 == ++v9 )
            {
              if ( !v12 )
                goto LABEL_23;
              *v12 = v8;
              --*(_DWORD *)(a3 + 32);
              ++*(_QWORD *)a3;
              goto LABEL_17;
            }
          }
          goto LABEL_6;
        }
LABEL_23:
        if ( v11 < *(_DWORD *)(a3 + 24) )
        {
          *(_DWORD *)(a3 + 28) = v11 + 1;
          *v10 = v8;
          ++*(_QWORD *)a3;
          goto LABEL_17;
        }
      }
      sub_16CCBA0(a3, *(_QWORD *)(v5 + 16));
      if ( v7 )
      {
LABEL_17:
        v5 += 32;
        sub_14D01A0(v8, (__int64)&v15, (__int64)v13);
        if ( v6 == v5 )
          break;
      }
      else
      {
LABEL_6:
        v5 += 32;
        if ( v6 == v5 )
          break;
      }
    }
  }
  sub_14D02F0((__int64)&v15, (__int64)v13, a3);
  if ( (_BYTE *)v13[0] != v14 )
    _libc_free(v13[0]);
  if ( v17 != v16 )
    _libc_free((unsigned __int64)v17);
}
