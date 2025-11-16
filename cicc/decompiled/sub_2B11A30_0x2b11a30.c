// Function: sub_2B11A30
// Address: 0x2b11a30
//
__int64 __fastcall sub_2B11A30(__int64 a1, char **a2, char **a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r13
  __int64 v10; // rcx
  char **v11; // rbx
  __int64 v12; // r15
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rdx
  char **v18; // rbx
  __int64 v19; // rdx
  __int64 v20; // r13
  __int64 i; // r15
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  char **v27; // rbx
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rcx
  __int64 v33; // [rsp+8h] [rbp-98h]
  __int64 v34; // [rsp+10h] [rbp-90h]
  __int64 v35; // [rsp+18h] [rbp-88h]
  __int64 v36; // [rsp+20h] [rbp-80h]
  __int64 v37; // [rsp+28h] [rbp-78h]
  char *v38; // [rsp+30h] [rbp-70h] BYREF
  __int64 v39; // [rsp+38h] [rbp-68h]
  _BYTE v40[96]; // [rsp+40h] [rbp-60h] BYREF

  v34 = a1;
  if ( (char **)a1 == a2 )
    return (__int64)a3;
  if ( a2 == a3 )
    return a1;
  v7 = a1 + (char *)a3 - (char *)a2;
  v33 = v7;
  v35 = ((__int64)a3 - a1) >> 6;
  v36 = ((__int64)a2 - a1) >> 6;
  if ( v36 != v35 - v36 )
  {
    while ( 1 )
    {
      v8 = v36;
      v37 = v35 - v36;
      if ( v36 >= v35 - v36 )
      {
        v18 = (char **)(v34 + (v35 << 6));
        v19 = (__int64)&v18[-8 * v37];
        v34 = v19;
        if ( v36 > 0 )
        {
          v20 = (__int64)&v18[-8 * v37];
          for ( i = 0; i != v36; ++i )
          {
            v20 -= 64;
            v38 = v40;
            v18 -= 8;
            v39 = 0x600000000LL;
            if ( *(_DWORD *)(v20 + 8) )
              sub_2B0F6D0((__int64)&v38, (char **)v20, v19, v8, a5, a6);
            sub_2B0F6D0(v20, v18, v19, v8, a5, a6);
            sub_2B0F6D0((__int64)v18, &v38, v22, v23, v24, v25);
            if ( v38 != v40 )
              _libc_free((unsigned __int64)v38);
          }
          v34 -= v36 << 6;
        }
        v36 = v35 % v37;
        if ( !(v35 % v37) )
          return v33;
      }
      else
      {
        v9 = v34;
        v10 = v36 << 6;
        v11 = (char **)(v34 + (v36 << 6));
        if ( v35 - v36 > 0 )
        {
          v12 = 0;
          do
          {
            v17 = *(unsigned int *)(v9 + 8);
            v38 = v40;
            v39 = 0x600000000LL;
            if ( (_DWORD)v17 )
              sub_2B0F6D0((__int64)&v38, (char **)v9, v17, v10, a5, a6);
            sub_2B0F6D0(v9, v11, v17, v10, a5, a6);
            sub_2B0F6D0((__int64)v11, &v38, v13, v14, v15, v16);
            if ( v38 != v40 )
              _libc_free((unsigned __int64)v38);
            v9 += 64;
            v11 += 8;
            ++v12;
          }
          while ( v37 != v12 );
          v34 += v37 << 6;
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
  v27 = a2;
  do
  {
    v38 = v40;
    v39 = 0x600000000LL;
    v32 = *(unsigned int *)(v33 + 8);
    if ( (_DWORD)v32 )
      sub_2B0F6D0((__int64)&v38, (char **)v33, v7, v32, a5, a6);
    sub_2B0F6D0(v33, v27, v7, v32, a5, a6);
    sub_2B0F6D0((__int64)v27, &v38, v28, v29, v30, v31);
    if ( v38 != v40 )
      _libc_free((unsigned __int64)v38);
    v33 += 64;
    v27 += 8;
  }
  while ( a2 != (char **)v33 );
  return v33;
}
