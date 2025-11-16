// Function: sub_908220
// Address: 0x908220
//
__int64 __fastcall sub_908220(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  const char *v5; // r8
  char *v6; // rax
  const char *v7; // r8
  size_t v8; // r9
  char *v9; // rdx
  char *v10; // rsi
  size_t v11; // rdx
  __int64 v12; // rsi
  __int64 result; // rax
  _BYTE *v14; // rbx
  _BYTE *v15; // r12
  _BYTE *v16; // rdi
  char *v17; // rax
  char *v18; // rdi
  size_t n; // [rsp+0h] [rbp-260h]
  const char *src; // [rsp+10h] [rbp-250h]
  int v23; // [rsp+30h] [rbp-230h] BYREF
  __int64 v24; // [rsp+38h] [rbp-228h]
  char *s[2]; // [rsp+40h] [rbp-220h] BYREF
  _QWORD v26[2]; // [rsp+50h] [rbp-210h] BYREF
  _QWORD v27[14]; // [rsp+60h] [rbp-200h] BYREF
  _QWORD *v28; // [rsp+D0h] [rbp-190h]
  __int64 v29; // [rsp+D8h] [rbp-188h]
  _QWORD v30[3]; // [rsp+E0h] [rbp-180h] BYREF
  int v31; // [rsp+F8h] [rbp-168h]
  _QWORD *v32; // [rsp+100h] [rbp-160h]
  __int64 v33; // [rsp+108h] [rbp-158h]
  _QWORD v34[2]; // [rsp+110h] [rbp-150h] BYREF
  _QWORD *v35; // [rsp+120h] [rbp-140h]
  __int64 v36; // [rsp+128h] [rbp-138h]
  _QWORD v37[2]; // [rsp+130h] [rbp-130h] BYREF
  __int64 v38; // [rsp+140h] [rbp-120h]
  __int64 v39; // [rsp+148h] [rbp-118h]
  __int64 v40; // [rsp+150h] [rbp-110h]
  _BYTE *v41; // [rsp+158h] [rbp-108h]
  __int64 v42; // [rsp+160h] [rbp-100h]
  _BYTE v43[248]; // [rsp+168h] [rbp-F8h] BYREF

  v28 = v30;
  v27[12] = 0;
  v27[13] = 0;
  v29 = 0;
  LOBYTE(v30[0]) = 0;
  v30[2] = 0;
  v31 = 0;
  v32 = v34;
  v33 = 0;
  LOBYTE(v34[0]) = 0;
  v35 = v37;
  v36 = 0;
  LOBYTE(v37[0]) = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v41 = v43;
  v42 = 0x400000000LL;
  v23 = 0;
  v4 = sub_2241E40();
  v5 = *(const char **)(a3 + 32);
  v24 = v4;
  s[0] = (char *)v26;
  if ( !v5 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  src = v5;
  v6 = (char *)strlen(v5);
  v7 = src;
  v27[0] = v6;
  v8 = (size_t)v6;
  if ( (unsigned __int64)v6 > 0xF )
  {
    n = (size_t)v6;
    v17 = (char *)sub_22409D0(s, v27, 0);
    v7 = src;
    v8 = n;
    s[0] = v17;
    v18 = v17;
    v26[0] = v27[0];
  }
  else
  {
    if ( v6 == (char *)1 )
    {
      LOBYTE(v26[0]) = *src;
      v9 = (char *)v26;
      goto LABEL_5;
    }
    if ( !v6 )
    {
      v9 = (char *)v26;
      goto LABEL_5;
    }
    v18 = (char *)v26;
  }
  memcpy(v18, v7, v8);
  v6 = (char *)v27[0];
  v9 = s[0];
LABEL_5:
  s[1] = v6;
  v6[(_QWORD)v9] = 0;
  v10 = s[0];
  v11 = 0;
  if ( s[0] )
  {
    v10 = s[0];
    v11 = strlen(s[0]);
  }
  sub_CB7060(v27, v10, v11, &v23, 0);
  v12 = a1;
  sub_CB6200(v27, a1, a2);
  sub_CB5B00(v27);
  if ( (_QWORD *)s[0] != v26 )
  {
    v12 = v26[0] + 1LL;
    j_j___libc_free_0(s[0], v26[0] + 1LL);
  }
  result = (unsigned int)v42;
  v14 = v41;
  v15 = &v41[48 * (unsigned int)v42];
  if ( v41 != v15 )
  {
    do
    {
      v15 -= 48;
      v16 = (_BYTE *)*((_QWORD *)v15 + 2);
      result = (__int64)(v15 + 32);
      if ( v16 != v15 + 32 )
      {
        v12 = *((_QWORD *)v15 + 4) + 1LL;
        result = j_j___libc_free_0(v16, v12);
      }
    }
    while ( v14 != v15 );
    v15 = v41;
  }
  if ( v15 != v43 )
    result = _libc_free(v15, v12);
  if ( v38 )
    result = j_j___libc_free_0(v38, v40 - v38);
  if ( v35 != v37 )
    result = j_j___libc_free_0(v35, v37[0] + 1LL);
  if ( v32 != v34 )
    result = j_j___libc_free_0(v32, v34[0] + 1LL);
  if ( v28 != v30 )
    return j_j___libc_free_0(v28, v30[0] + 1LL);
  return result;
}
