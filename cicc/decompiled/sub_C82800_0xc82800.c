// Function: sub_C82800
// Address: 0xc82800
//
__int64 __fastcall sub_C82800(_QWORD *a1)
{
  char *v2; // rax
  const char *v3; // r12
  size_t v4; // r12
  const char *v5; // rdi
  size_t v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  int *v10; // r12
  size_t v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  const char *v17; // rsi
  __int64 v18; // r14
  __int64 v19; // rdx
  __int64 v20; // r15
  __int64 v21; // rdx
  char *v22; // rdi
  __int64 v23; // rcx
  __int64 v24; // r8
  size_t v25; // r13
  __int64 v26; // rax
  size_t v27; // rdx
  char *v28; // [rsp+0h] [rbp-160h] BYREF
  __int16 v29; // [rsp+20h] [rbp-140h]
  const char *v30; // [rsp+30h] [rbp-130h] BYREF
  __int16 v31; // [rsp+50h] [rbp-110h]
  char *v32; // [rsp+60h] [rbp-100h] BYREF
  char v33; // [rsp+80h] [rbp-E0h]
  char v34; // [rsp+81h] [rbp-DFh]
  _OWORD v35[2]; // [rsp+90h] [rbp-D0h] BYREF
  __int128 v36; // [rsp+B0h] [rbp-B0h]
  __int128 v37; // [rsp+C0h] [rbp-A0h]
  __int64 v38; // [rsp+D0h] [rbp-90h]
  _OWORD v39[2]; // [rsp+E0h] [rbp-80h] BYREF
  __int128 v40; // [rsp+100h] [rbp-60h]
  __int128 v41; // [rsp+110h] [rbp-50h]
  __int64 v42; // [rsp+120h] [rbp-40h]

  a1[1] = 0;
  v2 = getenv("PWD");
  v38 = 0;
  v36 = 0;
  v40 = 0;
  HIDWORD(v36) = 0xFFFF;
  v42 = 0;
  HIDWORD(v40) = 0xFFFF;
  memset(v35, 0, sizeof(v35));
  v37 = 0;
  memset(v39, 0, sizeof(v39));
  v41 = 0;
  if ( !v2 )
    goto LABEL_8;
  v3 = v2;
  v29 = 257;
  if ( *v2 )
  {
    v28 = v2;
    LOBYTE(v29) = 3;
  }
  if ( !(unsigned __int8)sub_C81DB0((const char **)&v28, 0) )
    goto LABEL_8;
  v31 = 257;
  if ( *v3 )
  {
    v30 = v3;
    LOBYTE(v31) = 3;
  }
  if ( !(unsigned int)sub_C826E0((__int64)&v30, (__int64)v35, 1)
    && (v34 = 1, v17 = (const char *)v39, v32 = ".", v33 = 3, !(unsigned int)sub_C826E0((__int64)&v32, (__int64)v39, 1))
    && (v18 = sub_C82290((__int64)v39), v20 = v19, sub_C82290((__int64)v35) == v18)
    && v21 == v20 )
  {
    v22 = (char *)v3;
    v25 = strlen(v3);
    v26 = a1[1];
    v27 = v25 + v26;
    if ( v25 + v26 > a1[2] )
    {
      v17 = (const char *)(a1 + 3);
      v22 = (char *)a1;
      sub_C8D290(a1, a1 + 3, v27, 1);
      v26 = a1[1];
    }
    if ( v25 )
    {
      v17 = v3;
      v22 = (char *)(*a1 + v26);
      memcpy(v22, v3, v25);
      v26 = a1[1];
    }
    a1[1] = v26 + v25;
    sub_2241E40(v22, v17, v27, v23, v24);
    return 0;
  }
  else
  {
LABEL_8:
    if ( a1[1] != 4096 )
    {
      if ( a1[1] <= 0x1000u && a1[2] <= 0xFFFu )
        sub_C8D290(a1, a1 + 3, 4096, 1);
      a1[1] = 4096;
    }
    v4 = 4096;
LABEL_14:
    v5 = (const char *)*a1;
    v6 = v4;
    if ( getcwd((char *)*a1, v4) )
    {
LABEL_21:
      v12 = *a1;
      a1[1] = strlen((const char *)*a1);
      sub_2241E40(v12, v6, v13, v14, v15);
      return 0;
    }
    else
    {
      while ( 1 )
      {
        v10 = __errno_location();
        if ( *v10 != 12 )
          break;
        v11 = a1[2];
        v4 = 2 * v11;
        if ( 2 * v11 == a1[1] )
          goto LABEL_14;
        if ( 2 * v11 >= a1[1] && v4 > v11 )
          sub_C8D290(a1, a1 + 3, 2 * v11, 1);
        a1[1] = v4;
        v5 = (const char *)*a1;
        v6 = v4;
        if ( getcwd((char *)*a1, v4) )
          goto LABEL_21;
      }
      a1[1] = 0;
      sub_2241E50(v5, v6, v7, v8, v9);
      return (unsigned int)*v10;
    }
  }
}
