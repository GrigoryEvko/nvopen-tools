// Function: sub_C6C520
// Address: 0xc6c520
//
char **__fastcall sub_C6C520(char **a1, __int64 *a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rax
  unsigned __int64 i; // rbx
  char *v8; // rsi
  char *v9; // r12
  unsigned __int64 v10; // rax
  char *v11; // r15
  __int64 v12; // r8
  char *j; // rbx
  __int64 v14; // rcx
  size_t v15; // r13
  size_t v16; // r12
  size_t v17; // rdx
  int v18; // eax
  char *v19; // [rsp+8h] [rbp-78h]
  __int64 v20; // [rsp+10h] [rbp-70h]
  __int64 v21; // [rsp+18h] [rbp-68h]
  __int64 v22; // [rsp+28h] [rbp-58h] BYREF
  _QWORD v23[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v24; // [rsp+40h] [rbp-40h]
  __int64 v25; // [rsp+48h] [rbp-38h]

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  if ( *((_DWORD *)a2 + 4) )
  {
    v3 = *((unsigned int *)a2 + 6);
    v4 = a2[1];
    v5 = *a2;
    v23[0] = a2;
    v24 = v4;
    v23[1] = v5;
    v25 = v4 + (v3 << 6);
    sub_C6B5D0((__int64)v23);
    v6 = v24;
    for ( i = a2[1] + ((unsigned __int64)*((unsigned int *)a2 + 6) << 6); i != v24; v6 = v24 )
    {
      v22 = v6;
      v8 = a1[1];
      if ( v8 == a1[2] )
      {
        sub_C6C390((__int64)a1, v8, &v22);
      }
      else
      {
        if ( v8 )
        {
          *(_QWORD *)v8 = v6;
          v8 = a1[1];
        }
        a1[1] = v8 + 8;
      }
      v24 += 64;
      sub_C6B5D0((__int64)v23);
    }
    v9 = *a1;
    v19 = a1[1];
    if ( v19 != *a1 )
    {
      _BitScanReverse64(&v10, (v19 - v9) >> 3);
      sub_C695B0(v9, v19, 2LL * (int)(63 - (v10 ^ 0x3F)));
      if ( v19 - v9 <= 128 )
      {
        sub_C68ED0(v9, v19);
      }
      else
      {
        v11 = v9 + 128;
        sub_C68ED0(v9, v9 + 128);
        if ( v9 + 128 != v19 )
        {
          do
          {
            v12 = *(_QWORD *)v11;
            for ( j = v11; ; j -= 8 )
            {
              while ( 1 )
              {
                v14 = *((_QWORD *)j - 1);
                v15 = *(_QWORD *)(v12 + 16);
                v16 = *(_QWORD *)(v14 + 16);
                v17 = v16;
                if ( v15 <= v16 )
                  v17 = *(_QWORD *)(v12 + 16);
                if ( !v17 )
                  break;
                v20 = *((_QWORD *)j - 1);
                v21 = v12;
                v18 = memcmp(*(const void **)(v12 + 8), *(const void **)(v14 + 8), v17);
                v12 = v21;
                v14 = v20;
                if ( !v18 )
                  break;
                if ( v18 >= 0 )
                  goto LABEL_15;
                *(_QWORD *)j = v20;
                j -= 8;
              }
              if ( v15 == v16 || v15 >= v16 )
                break;
              *(_QWORD *)j = v14;
            }
LABEL_15:
            *(_QWORD *)j = v12;
            v11 += 8;
          }
          while ( v19 != v11 );
        }
      }
    }
  }
  return a1;
}
