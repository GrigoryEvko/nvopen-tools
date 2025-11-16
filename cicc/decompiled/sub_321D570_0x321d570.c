// Function: sub_321D570
// Address: 0x321d570
//
unsigned __int64 __fastcall sub_321D570(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  char **v8; // r12
  __int64 v9; // rbx
  char **v10; // r14
  __int64 v11; // rcx
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // r15
  char **v17; // rax
  __int64 v18; // rdx
  char **v19; // r15
  char **v21; // rbx
  __int64 v22; // rdx
  __int64 v23; // rcx
  char **v24; // rdi
  __int64 v25; // rbx
  __int64 v26; // rcx
  __int64 v27; // rdx
  unsigned __int64 v28; // r12
  __int64 v29; // rbx
  char **v30; // r15
  __int64 v31; // rax
  char **v32; // rsi
  __int64 v33; // rdi
  unsigned __int64 v34; // rdi
  char **v35; // [rsp+8h] [rbp-38h]
  __int64 v36; // [rsp+8h] [rbp-38h]

  v7 = *(unsigned int *)(a1 + 24);
  v8 = *(char ***)(a1 + 16);
  v9 = 80 * v7;
  v10 = &v8[10 * v7];
  v11 = (__int64)v10;
  if ( v8 != v10 )
  {
    v12 = (unsigned __int64)&v8[10 * v7];
    _BitScanReverse64(&v13, 0xCCCCCCCCCCCCCCCDLL * (v9 >> 4));
    sub_321CCA0(*(_QWORD *)(a1 + 16), v12, 2LL * (int)(63 - (v13 ^ 0x3F)), (__int64)v10, a5, a6);
    if ( (unsigned __int64)v9 > 0x500 )
    {
      v21 = v8 + 160;
      sub_321A850(v8, v8 + 160);
      if ( v10 != v8 + 160 )
      {
        do
        {
          v24 = v21;
          v21 += 10;
          sub_3219F60(v24, (__int64)(v8 + 160), v22, v23, v14, v15);
        }
        while ( v10 != v21 );
      }
    }
    else
    {
      sub_321A850(v8, v10);
    }
    v11 = *(_QWORD *)(a1 + 16);
    v10 = (char **)v11;
    v16 = 80LL * *(unsigned int *)(a1 + 24);
    v17 = (char **)(v11 + v16);
    if ( v11 != v11 + v16 )
    {
      while ( 1 )
      {
        v19 = v10;
        v10 += 10;
        if ( v17 == v10 )
          break;
        v18 = (__int64)*(v10 - 10);
        if ( (char *)v18 == *v10 )
        {
          if ( v17 == v19 )
          {
            v10 = v17;
          }
          else
          {
            v25 = (__int64)(v19 + 20);
            if ( v17 != v19 + 20 )
            {
              while ( 1 )
              {
                v26 = *(_QWORD *)v25;
                if ( *(_QWORD *)v25 != v18 )
                {
                  v19[10] = (char *)v26;
                  v35 = v17;
                  sub_3218940((__int64)(v19 + 11), (char **)(v25 + 8), v18, v26, v14, v15);
                  v17 = v35;
                  *((_BYTE *)v19 + 152) = *(_BYTE *)(v25 + 72);
                  v19 += 10;
                }
                v25 += 80;
                if ( v17 == (char **)v25 )
                  break;
                v18 = (__int64)*v19;
              }
              v11 = *(_QWORD *)(a1 + 16);
              v10 = v19 + 10;
              v27 = 0xCCCCCCCCCCCCCCCDLL;
              v25 = v11 + 80LL * *(unsigned int *)(a1 + 24);
              v36 = v25 - (_QWORD)v17;
              v28 = 0xCCCCCCCCCCCCCCCDLL * ((v25 - (__int64)v17) >> 4);
              if ( v25 - (__int64)v17 > 0 )
              {
                v29 = (__int64)(v19 + 11);
                v30 = v17 + 1;
                do
                {
                  v31 = (__int64)*(v30 - 1);
                  v32 = v30;
                  v33 = v29;
                  v30 += 10;
                  v29 += 80;
                  *(_QWORD *)(v29 - 88) = v31;
                  sub_3218940(v33, v32, v27, v11, v14, v15);
                  *(_BYTE *)(v29 - 16) = *((_BYTE *)v30 - 16);
                  --v28;
                }
                while ( v28 );
                v11 = *(_QWORD *)(a1 + 16);
                v10 = (char **)((char *)v10 + v36);
                v25 = v11 + 80LL * *(unsigned int *)(a1 + 24);
              }
            }
            if ( v10 != (char **)v25 )
            {
              do
              {
                v25 -= 80;
                v34 = *(_QWORD *)(v25 + 8);
                if ( v34 != v25 + 24 )
                  _libc_free(v34);
              }
              while ( v10 != (char **)v25 );
              v11 = *(_QWORD *)(a1 + 16);
            }
          }
          break;
        }
      }
    }
  }
  *(_DWORD *)(a1 + 24) = -858993459 * (((__int64)v10 - v11) >> 4);
  return 0xCCCCCCCCCCCCCCCDLL;
}
