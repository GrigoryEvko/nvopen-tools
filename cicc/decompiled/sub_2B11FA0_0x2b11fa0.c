// Function: sub_2B11FA0
// Address: 0x2b11fa0
//
__int64 __fastcall sub_2B11FA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v7; // r15
  char **v8; // r12
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rcx
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r12
  char **v19; // r15
  __int64 i; // r13
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rcx
  __int64 v26; // rsi
  char **v28; // r15
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // rdx
  char *v34; // rax
  char **v35; // r8
  __int64 v36; // rcx
  __int64 v37; // [rsp+8h] [rbp-98h]
  __int64 v38; // [rsp+10h] [rbp-90h]
  signed __int64 v39; // [rsp+18h] [rbp-88h]
  __int64 v40; // [rsp+20h] [rbp-80h]
  signed __int64 v41; // [rsp+28h] [rbp-78h]
  char *v42; // [rsp+30h] [rbp-70h] BYREF
  __int64 v43; // [rsp+38h] [rbp-68h]
  _BYTE v44[96]; // [rsp+40h] [rbp-60h] BYREF

  v38 = a1;
  if ( a1 == a2 )
    return a3;
  if ( a2 == a3 )
    return a1;
  v37 = a1 + a3 - a2;
  v39 = 0x8E38E38E38E38E39LL * ((a3 - a1) >> 3);
  v40 = 0x8E38E38E38E38E39LL * ((a2 - a1) >> 3);
  if ( v40 != v39 - v40 )
  {
    while ( 1 )
    {
      v41 = v39 - v40;
      if ( v40 >= v39 - v40 )
      {
        v16 = v38 + 72 * v39;
        v17 = 72 * v41;
        v38 = v16 - 72 * v41;
        if ( v40 > 0 )
        {
          v18 = v16 - 72 * v41 - 64;
          v19 = (char **)(v16 - 64);
          for ( i = 0; i != v40; ++i )
          {
            v25 = *(_QWORD *)(v18 - 8);
            v26 = (__int64)*(v19 - 1);
            v42 = v44;
            v43 = 0xC00000000LL;
            *(_QWORD *)(v18 - 8) = v26;
            *(v19 - 1) = (char *)v25;
            if ( *(_DWORD *)(v18 + 8) )
              sub_2B0D090((__int64)&v42, (char **)v18, v17, v25, a5, a6);
            sub_2B0D090(v18, v19, v17, v25, a5, a6);
            sub_2B0D090((__int64)v19, &v42, v21, v22, v23, v24);
            if ( v42 != v44 )
              _libc_free((unsigned __int64)v42);
            v18 -= 72;
            v19 -= 9;
          }
          v38 += -72 * v40;
        }
        v40 = v39 % v41;
        if ( !(v39 % v41) )
          return v37;
      }
      else
      {
        if ( v39 - v40 > 0 )
        {
          v6 = 0;
          v7 = v38 + 8;
          v8 = (char **)(v38 + 72 * v40 + 8);
          do
          {
            v13 = *(_QWORD *)(v7 - 8);
            v14 = (__int64)*(v8 - 1);
            v42 = v44;
            v43 = 0xC00000000LL;
            *(_QWORD *)(v7 - 8) = v14;
            *(v8 - 1) = (char *)v13;
            v15 = *(unsigned int *)(v7 + 8);
            if ( (_DWORD)v15 )
              sub_2B0D090((__int64)&v42, (char **)v7, v15, v13, a5, a6);
            sub_2B0D090(v7, v8, v15, v13, a5, a6);
            sub_2B0D090((__int64)v8, &v42, v9, v10, v11, v12);
            if ( v42 != v44 )
              _libc_free((unsigned __int64)v42);
            ++v6;
            v7 += 72;
            v8 += 9;
          }
          while ( v41 != v6 );
          v38 += 72 * v41;
        }
        if ( !(v39 % v40) )
          return v37;
        v41 = v40;
        v40 -= v39 % v40;
      }
      v39 = v41;
    }
  }
  v37 = a1;
  v28 = (char **)(a2 + 8);
  do
  {
    v33 = (__int64)*(v28 - 1);
    v42 = v44;
    v43 = 0xC00000000LL;
    v34 = *(char **)v37;
    v35 = (char **)(v37 + 8);
    *(_QWORD *)v37 = v33;
    *(v28 - 1) = v34;
    v36 = *(unsigned int *)(v37 + 16);
    if ( (_DWORD)v36 )
    {
      sub_2B0D090((__int64)&v42, v35, v33, v36, (__int64)v35, a6);
      v35 = (char **)(v37 + 8);
    }
    sub_2B0D090((__int64)v35, v28, v33, v36, (__int64)v35, a6);
    sub_2B0D090((__int64)v28, &v42, v29, v30, v31, v32);
    if ( v42 != v44 )
      _libc_free((unsigned __int64)v42);
    v37 += 72;
    v28 += 9;
  }
  while ( a2 != v37 );
  return v37;
}
