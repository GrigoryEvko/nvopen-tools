// Function: sub_27AD1C0
// Address: 0x27ad1c0
//
__int64 __fastcall sub_27AD1C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v7; // r15
  char **v8; // r12
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // r8
  __int64 v12; // r9
  int v13; // ecx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r12
  char **v18; // r15
  __int64 i; // r13
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // r8
  __int64 v23; // r9
  int v24; // ecx
  char **v26; // r12
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // [rsp+8h] [rbp-A8h]
  __int64 v34; // [rsp+10h] [rbp-A0h]
  signed __int64 v35; // [rsp+18h] [rbp-98h]
  __int64 v36; // [rsp+20h] [rbp-90h]
  signed __int64 v37; // [rsp+28h] [rbp-88h]
  int v38; // [rsp+30h] [rbp-80h]
  int v39; // [rsp+30h] [rbp-80h]
  int v40; // [rsp+30h] [rbp-80h]
  int v41; // [rsp+34h] [rbp-7Ch]
  int v42; // [rsp+34h] [rbp-7Ch]
  int v43; // [rsp+34h] [rbp-7Ch]
  int v44; // [rsp+38h] [rbp-78h]
  int v45; // [rsp+38h] [rbp-78h]
  int v46; // [rsp+38h] [rbp-78h]
  int v47; // [rsp+3Ch] [rbp-74h]
  int v48; // [rsp+3Ch] [rbp-74h]
  int v49; // [rsp+3Ch] [rbp-74h]
  unsigned int v50; // [rsp+40h] [rbp-70h]
  unsigned int v51; // [rsp+40h] [rbp-70h]
  int v52; // [rsp+40h] [rbp-70h]
  char *v53; // [rsp+48h] [rbp-68h] BYREF
  __int64 v54; // [rsp+50h] [rbp-60h]
  _BYTE v55[88]; // [rsp+58h] [rbp-58h] BYREF

  v34 = a1;
  if ( a1 == a2 )
    return a3;
  if ( a2 == a3 )
    return a1;
  v33 = a1 + a3 - a2;
  v35 = 0x8E38E38E38E38E39LL * ((a3 - a1) >> 3);
  v36 = 0x8E38E38E38E38E39LL * ((a2 - a1) >> 3);
  if ( v36 != v35 - v36 )
  {
    while ( 1 )
    {
      v37 = v35 - v36;
      if ( v36 >= v35 - v36 )
      {
        v15 = v34 + 72 * v35;
        v16 = 72 * v37;
        v34 = v15 - 72 * v37;
        if ( v36 > 0 )
        {
          v17 = v15 - 72 * v37 - 48;
          v18 = (char **)(v15 - 48);
          for ( i = 0; i != v36; ++i )
          {
            v24 = *(_DWORD *)(v17 - 24);
            v53 = v55;
            v54 = 0x400000000LL;
            v39 = v24;
            v42 = *(_DWORD *)(v17 - 20);
            v45 = *(_DWORD *)(v17 - 16);
            v48 = *(_DWORD *)(v17 - 12);
            v51 = *(_DWORD *)(v17 - 8);
            if ( *(_DWORD *)(v17 + 8) )
              sub_27AC070((__int64)&v53, (char **)v17, v16, *(unsigned int *)(v17 - 8), a5, a6);
            *(_DWORD *)(v17 - 24) = *((_DWORD *)v18 - 6);
            *(_DWORD *)(v17 - 20) = *((_DWORD *)v18 - 5);
            *(_DWORD *)(v17 - 16) = *((_DWORD *)v18 - 4);
            *(_DWORD *)(v17 - 12) = *((_DWORD *)v18 - 3);
            v20 = *((unsigned int *)v18 - 2);
            *(_DWORD *)(v17 - 8) = v20;
            sub_27AC070(v17, v18, v16, v20, a5, a6);
            *((_DWORD *)v18 - 6) = v39;
            *((_DWORD *)v18 - 5) = v42;
            *((_DWORD *)v18 - 4) = v45;
            *((_DWORD *)v18 - 3) = v48;
            *((_DWORD *)v18 - 2) = v51;
            sub_27AC070((__int64)v18, &v53, v21, v51, v22, v23);
            if ( v53 != v55 )
              _libc_free((unsigned __int64)v53);
            v17 -= 72;
            v18 -= 9;
          }
          v34 += -72 * v36;
        }
        v36 = v35 % v37;
        if ( !(v35 % v37) )
          return v33;
      }
      else
      {
        if ( v35 - v36 > 0 )
        {
          v6 = 0;
          v7 = v34 + 24;
          v8 = (char **)(v34 + 72 * v36 + 24);
          do
          {
            v13 = *(_DWORD *)(v7 - 24);
            v14 = *(unsigned int *)(v7 + 8);
            v53 = v55;
            v54 = 0x400000000LL;
            v38 = v13;
            v41 = *(_DWORD *)(v7 - 20);
            v44 = *(_DWORD *)(v7 - 16);
            v47 = *(_DWORD *)(v7 - 12);
            v50 = *(_DWORD *)(v7 - 8);
            if ( (_DWORD)v14 )
              sub_27AC070((__int64)&v53, (char **)v7, v14, *(unsigned int *)(v7 - 8), a5, a6);
            *(_DWORD *)(v7 - 24) = *((_DWORD *)v8 - 6);
            *(_DWORD *)(v7 - 20) = *((_DWORD *)v8 - 5);
            *(_DWORD *)(v7 - 16) = *((_DWORD *)v8 - 4);
            *(_DWORD *)(v7 - 12) = *((_DWORD *)v8 - 3);
            v9 = *((unsigned int *)v8 - 2);
            *(_DWORD *)(v7 - 8) = v9;
            sub_27AC070(v7, v8, v14, v9, a5, a6);
            *((_DWORD *)v8 - 6) = v38;
            *((_DWORD *)v8 - 5) = v41;
            *((_DWORD *)v8 - 4) = v44;
            *((_DWORD *)v8 - 3) = v47;
            *((_DWORD *)v8 - 2) = v50;
            sub_27AC070((__int64)v8, &v53, v10, v50, v11, v12);
            if ( v53 != v55 )
              _libc_free((unsigned __int64)v53);
            ++v6;
            v7 += 72;
            v8 += 9;
          }
          while ( v37 != v6 );
          v34 += 72 * v37;
        }
        if ( !(v35 % v36) )
          return v33;
        v37 = v36;
        v36 -= v35 % v36;
      }
      v35 = v37;
    }
  }
  v33 = a1;
  v26 = (char **)(a2 + 24);
  do
  {
    v53 = v55;
    v54 = 0x400000000LL;
    v31 = *(unsigned int *)(v33 + 32);
    v32 = v33 + 24;
    v40 = *(_DWORD *)v33;
    v43 = *(_DWORD *)(v33 + 4);
    v46 = *(_DWORD *)(v33 + 8);
    v49 = *(_DWORD *)(v33 + 12);
    v52 = *(_DWORD *)(v33 + 16);
    if ( (_DWORD)v31 )
    {
      sub_27AC070((__int64)&v53, (char **)(v33 + 24), v33, v31, v32, a6);
      v32 = v33 + 24;
    }
    *(_DWORD *)v33 = *((_DWORD *)v26 - 6);
    *(_DWORD *)(v33 + 4) = *((_DWORD *)v26 - 5);
    *(_DWORD *)(v33 + 8) = *((_DWORD *)v26 - 4);
    *(_DWORD *)(v33 + 12) = *((_DWORD *)v26 - 3);
    *(_DWORD *)(v33 + 16) = *((_DWORD *)v26 - 2);
    sub_27AC070(v32, v26, v33, v31, v32, a6);
    *((_DWORD *)v26 - 6) = v40;
    *((_DWORD *)v26 - 5) = v43;
    *((_DWORD *)v26 - 4) = v46;
    *((_DWORD *)v26 - 3) = v49;
    *((_DWORD *)v26 - 2) = v52;
    sub_27AC070((__int64)v26, &v53, v27, v28, v29, v30);
    if ( v53 != v55 )
      _libc_free((unsigned __int64)v53);
    v33 += 72;
    v26 += 9;
  }
  while ( a2 != v33 );
  return v33;
}
