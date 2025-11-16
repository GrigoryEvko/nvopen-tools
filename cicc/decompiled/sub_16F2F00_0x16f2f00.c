// Function: sub_16F2F00
// Address: 0x16f2f00
//
char **__fastcall sub_16F2F00(char **a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rcx
  int v12; // r8d
  __int64 v13; // rax
  unsigned __int64 i; // rbx
  char *v15; // rsi
  char *v16; // r12
  unsigned __int64 v17; // rax
  char *v18; // r15
  __int64 v19; // r8
  char *j; // rbx
  int v21; // eax
  size_t v22; // rcx
  __int64 v23; // r13
  size_t v24; // r12
  const void *v25; // rdi
  const void *v26; // rsi
  char *v27; // [rsp+8h] [rbp-78h]
  size_t v28; // [rsp+10h] [rbp-70h]
  __int64 v29; // [rsp+10h] [rbp-70h]
  __int64 v30; // [rsp+18h] [rbp-68h]
  size_t v31; // [rsp+18h] [rbp-68h]
  __int64 v32; // [rsp+28h] [rbp-58h] BYREF
  _QWORD v33[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v34; // [rsp+40h] [rbp-40h]
  __int64 v35; // [rsp+48h] [rbp-38h]

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  if ( *(_DWORD *)(a2 + 16) )
  {
    v7 = *(unsigned int *)(a2 + 24);
    v8 = *(_QWORD *)(a2 + 8);
    v9 = *(_QWORD *)a2;
    v33[0] = a2;
    v34 = v8;
    v33[1] = v9;
    v35 = v8 + (v7 << 6);
    sub_16F2750((__int64)v33, a2, v8, v9, a5);
    v13 = v34;
    for ( i = *(_QWORD *)(a2 + 8) + ((unsigned __int64)*(unsigned int *)(a2 + 24) << 6); i != v34; v13 = v34 )
    {
      v32 = v13;
      v15 = a1[1];
      if ( v15 == a1[2] )
      {
        sub_16F2D70((__int64)a1, v15, &v32);
      }
      else
      {
        if ( v15 )
        {
          *(_QWORD *)v15 = v13;
          v15 = a1[1];
        }
        v15 += 8;
        a1[1] = v15;
      }
      v34 += 64;
      sub_16F2750((__int64)v33, (__int64)v15, v10, v11, v12);
    }
    v16 = *a1;
    v27 = a1[1];
    if ( v27 != *a1 )
    {
      _BitScanReverse64(&v17, (v27 - v16) >> 3);
      sub_16F1CC0(v16, v27, 2LL * (int)(63 - (v17 ^ 0x3F)));
      if ( v27 - v16 <= 128 )
      {
        sub_16F15D0(v16, v27);
      }
      else
      {
        v18 = v16 + 128;
        sub_16F15D0(v16, v16 + 128);
        if ( v16 + 128 != v27 )
        {
          do
          {
            v19 = *(_QWORD *)v18;
            for ( j = v18; ; j -= 8 )
            {
              v23 = *((_QWORD *)j - 1);
              v24 = *(_QWORD *)(v19 + 16);
              v25 = *(const void **)(v19 + 8);
              v22 = *(_QWORD *)(v23 + 16);
              v26 = *(const void **)(v23 + 8);
              if ( v24 > v22 )
                break;
              if ( v24 )
              {
                v28 = *(_QWORD *)(v23 + 16);
                v30 = v19;
                v21 = memcmp(v25, v26, *(_QWORD *)(v19 + 16));
                v19 = v30;
                v22 = v28;
                if ( v21 )
                  goto LABEL_23;
              }
              if ( v24 == v22 )
                goto LABEL_24;
LABEL_18:
              if ( v24 >= v22 )
                goto LABEL_24;
LABEL_19:
              *(_QWORD *)j = v23;
            }
            if ( !v22 )
              goto LABEL_24;
            v29 = v19;
            v31 = *(_QWORD *)(v23 + 16);
            v21 = memcmp(v25, v26, v31);
            v22 = v31;
            v19 = v29;
            if ( !v21 )
              goto LABEL_18;
LABEL_23:
            if ( v21 < 0 )
              goto LABEL_19;
LABEL_24:
            *(_QWORD *)j = v19;
            v18 += 8;
          }
          while ( v27 != v18 );
        }
      }
    }
  }
  return a1;
}
