// Function: sub_CFDFC0
// Address: 0xcfdfc0
//
__int64 __fastcall sub_CFDFC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  _QWORD *v6; // rcx
  __int64 v8; // r13
  __int64 result; // rax
  __int64 v10; // rbx
  __int64 j; // r15
  __int64 v12; // r12
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rsi
  int v15; // eax
  unsigned __int64 v16; // r12
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v19; // rbx
  __int64 k; // r12
  __int64 v21; // rsi
  __int64 v22; // rdi
  char *v23; // r12
  _QWORD *v24; // [rsp+8h] [rbp-68h]
  __int64 i; // [rsp+10h] [rbp-60h]
  _QWORD *v26; // [rsp+18h] [rbp-58h]
  __int64 v27; // [rsp+18h] [rbp-58h]
  _QWORD *v28; // [rsp+18h] [rbp-58h]
  _QWORD *v29; // [rsp+18h] [rbp-58h]
  _QWORD *v30; // [rsp+18h] [rbp-58h]
  _QWORD v31[2]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v32; // [rsp+30h] [rbp-40h]
  int v33; // [rsp+38h] [rbp-38h]

  v6 = v31;
  v8 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  result = *(_QWORD *)a1 + 72LL;
  for ( i = result; i != v8; v8 = *(_QWORD *)(v8 + 8) )
  {
    if ( !v8 )
      BUG();
    v10 = *(_QWORD *)(v8 + 32);
    for ( j = v8 + 24; j != v10; v6 = v28 )
    {
      while ( 1 )
      {
        if ( !v10 )
          BUG();
        if ( *(_BYTE *)(v10 - 24) == 85 )
        {
          result = *(_QWORD *)(v10 - 56);
          if ( result )
          {
            if ( !*(_BYTE *)result )
            {
              a3 = *(_QWORD *)(v10 + 56);
              if ( *(_QWORD *)(result + 24) == a3
                && (*(_BYTE *)(result + 33) & 0x20) != 0
                && *(_DWORD *)(result + 36) == 11 )
              {
                v31[0] = 4;
                v31[1] = 0;
                v32 = v10 - 24;
                if ( v10 != -4072 && v10 != -8168 )
                {
                  v26 = v6;
                  sub_BD73F0((__int64)v6);
                  v6 = v26;
                }
                v12 = *(unsigned int *)(a1 + 24);
                v13 = *(unsigned int *)(a1 + 28);
                v33 = -1;
                a5 = (__int64)v6;
                v14 = *(_QWORD *)(a1 + 16);
                a6 = (__int64 *)(v12 + 1);
                v15 = v12;
                if ( v12 + 1 > v13 )
                {
                  v22 = a1 + 16;
                  if ( v14 > (unsigned __int64)v6 || (unsigned __int64)v6 >= v14 + 32 * v12 )
                  {
                    v30 = v6;
                    sub_CFC2E0(v22, (unsigned __int64)a6, a3, (__int64)v6, (__int64)v6, (__int64)a6);
                    v12 = *(unsigned int *)(a1 + 24);
                    v6 = v30;
                    v14 = *(_QWORD *)(a1 + 16);
                    v15 = *(_DWORD *)(a1 + 24);
                    a5 = (__int64)v30;
                  }
                  else
                  {
                    v29 = v6;
                    v23 = (char *)v6 - v14;
                    sub_CFC2E0(v22, (unsigned __int64)a6, a3, (__int64)v6, (__int64)v6, (__int64)a6);
                    v14 = *(_QWORD *)(a1 + 16);
                    v6 = v29;
                    a5 = (__int64)&v23[v14];
                    v12 = *(unsigned int *)(a1 + 24);
                    v15 = *(_DWORD *)(a1 + 24);
                  }
                }
                v16 = v14 + 32 * v12;
                if ( v16 )
                {
                  *(_QWORD *)v16 = 4;
                  v17 = *(_QWORD *)(a5 + 16);
                  *(_QWORD *)(v16 + 8) = 0;
                  *(_QWORD *)(v16 + 16) = v17;
                  if ( v17 != 0 && v17 != -4096 && v17 != -8192 )
                  {
                    v24 = v6;
                    v27 = a5;
                    sub_BD6050((unsigned __int64 *)v16, *(_QWORD *)a5 & 0xFFFFFFFFFFFFFFF8LL);
                    v6 = v24;
                    a5 = v27;
                  }
                  *(_DWORD *)(v16 + 24) = *(_DWORD *)(a5 + 24);
                  v15 = *(_DWORD *)(a1 + 24);
                }
                *(_DWORD *)(a1 + 24) = v15 + 1;
                result = v32;
                if ( v32 != -4096 && v32 != 0 && v32 != -8192 )
                  break;
              }
            }
          }
        }
        v10 = *(_QWORD *)(v10 + 8);
        if ( j == v10 )
          goto LABEL_25;
      }
      v28 = v6;
      result = sub_BD60C0(v6);
      v10 = *(_QWORD *)(v10 + 8);
    }
LABEL_25:
    ;
  }
  v18 = *(unsigned int *)(a1 + 24);
  v19 = *(_QWORD *)(a1 + 16);
  *(_BYTE *)(a1 + 192) = 1;
  for ( k = v19 + 32 * v18; v19 != k; result = sub_CFDBA0(a1, v21, a3, (__int64)v6, a5, a6) )
  {
    v21 = *(_QWORD *)(v19 + 16);
    v19 += 32;
  }
  return result;
}
