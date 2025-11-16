// Function: sub_1267CC0
// Address: 0x1267cc0
//
__int64 __fastcall sub_1267CC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rax
  const char *v7; // r8
  char *v8; // rax
  const char *v9; // r8
  size_t v10; // r9
  char *v11; // rdx
  char *v12; // rsi
  size_t v13; // rdx
  __int64 v14; // rsi
  __int64 result; // rax
  _BYTE *v16; // rbx
  _BYTE *v17; // r12
  _BYTE *v18; // rdi
  char *v19; // rax
  char *v20; // rdi
  size_t n; // [rsp+0h] [rbp-250h]
  const char *src; // [rsp+10h] [rbp-240h]
  int v25; // [rsp+30h] [rbp-220h] BYREF
  __int64 v26; // [rsp+38h] [rbp-218h]
  char *s[2]; // [rsp+40h] [rbp-210h] BYREF
  _QWORD v28[2]; // [rsp+50h] [rbp-200h] BYREF
  _QWORD v29[12]; // [rsp+60h] [rbp-1F0h] BYREF
  _QWORD *v30; // [rsp+C0h] [rbp-190h]
  __int64 v31; // [rsp+C8h] [rbp-188h]
  _QWORD v32[3]; // [rsp+D0h] [rbp-180h] BYREF
  int v33; // [rsp+E8h] [rbp-168h]
  _QWORD *v34; // [rsp+F0h] [rbp-160h]
  __int64 v35; // [rsp+F8h] [rbp-158h]
  _QWORD v36[2]; // [rsp+100h] [rbp-150h] BYREF
  _QWORD *v37; // [rsp+110h] [rbp-140h]
  __int64 v38; // [rsp+118h] [rbp-138h]
  _QWORD v39[2]; // [rsp+120h] [rbp-130h] BYREF
  __int64 v40; // [rsp+130h] [rbp-120h]
  __int64 v41; // [rsp+138h] [rbp-118h]
  __int64 v42; // [rsp+140h] [rbp-110h]
  _BYTE *v43; // [rsp+148h] [rbp-108h]
  __int64 v44; // [rsp+150h] [rbp-100h]
  _BYTE v45[248]; // [rsp+158h] [rbp-F8h] BYREF

  v30 = v32;
  v29[10] = 0;
  v29[11] = 0;
  v31 = 0;
  LOBYTE(v32[0]) = 0;
  v32[2] = 0;
  v33 = 0;
  v34 = v36;
  v35 = 0;
  LOBYTE(v36[0]) = 0;
  v37 = v39;
  v38 = 0;
  LOBYTE(v39[0]) = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = v45;
  v44 = 0x400000000LL;
  v25 = 0;
  v6 = sub_2241E40(a1, a2, a3, a4, a5);
  v7 = *(const char **)(a3 + 32);
  v26 = v6;
  s[0] = (char *)v28;
  if ( !v7 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  src = v7;
  v8 = (char *)strlen(v7);
  v9 = src;
  v29[0] = v8;
  v10 = (size_t)v8;
  if ( (unsigned __int64)v8 > 0xF )
  {
    n = (size_t)v8;
    v19 = (char *)sub_22409D0(s, v29, 0);
    v9 = src;
    v10 = n;
    s[0] = v19;
    v20 = v19;
    v28[0] = v29[0];
  }
  else
  {
    if ( v8 == (char *)1 )
    {
      LOBYTE(v28[0]) = *src;
      v11 = (char *)v28;
      goto LABEL_5;
    }
    if ( !v8 )
    {
      v11 = (char *)v28;
      goto LABEL_5;
    }
    v20 = (char *)v28;
  }
  memcpy(v20, v9, v10);
  v8 = (char *)v29[0];
  v11 = s[0];
LABEL_5:
  s[1] = v8;
  v8[(_QWORD)v11] = 0;
  v12 = s[0];
  v13 = 0;
  if ( s[0] )
  {
    v12 = s[0];
    v13 = strlen(s[0]);
  }
  sub_16E8AF0(v29, v12, v13, &v25, 0);
  v14 = a1;
  sub_16E7EE0(v29, a1, a2);
  sub_16E7C30(v29);
  if ( (_QWORD *)s[0] != v28 )
  {
    v14 = v28[0] + 1LL;
    j_j___libc_free_0(s[0], v28[0] + 1LL);
  }
  result = (unsigned int)v44;
  v16 = v43;
  v17 = &v43[48 * (unsigned int)v44];
  if ( v43 != v17 )
  {
    do
    {
      v17 -= 48;
      v18 = (_BYTE *)*((_QWORD *)v17 + 2);
      result = (__int64)(v17 + 32);
      if ( v18 != v17 + 32 )
      {
        v14 = *((_QWORD *)v17 + 4) + 1LL;
        result = j_j___libc_free_0(v18, v14);
      }
    }
    while ( v16 != v17 );
    v17 = v43;
  }
  if ( v17 != v45 )
    result = _libc_free(v17, v14);
  if ( v40 )
    result = j_j___libc_free_0(v40, v42 - v40);
  if ( v37 != v39 )
    result = j_j___libc_free_0(v37, v39[0] + 1LL);
  if ( v34 != v36 )
    result = j_j___libc_free_0(v34, v36[0] + 1LL);
  if ( v30 != v32 )
    return j_j___libc_free_0(v30, v32[0] + 1LL);
  return result;
}
