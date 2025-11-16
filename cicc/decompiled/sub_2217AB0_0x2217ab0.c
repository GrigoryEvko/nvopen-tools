// Function: sub_2217AB0
// Address: 0x2217ab0
//
_QWORD *__fastcall sub_2217AB0(_QWORD *a1, __int64 a2, _BYTE *a3, __int64 a4)
{
  const char *v6; // r13
  unsigned __int64 v7; // rbx
  const char *v8; // rbp
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rdx
  const char *v11; // r13
  __int64 v12; // r14
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  unsigned __int64 v16; // [rsp+0h] [rbp-78h]
  char *v17; // [rsp+8h] [rbp-70h]
  char *s[2]; // [rsp+20h] [rbp-58h] BYREF
  _BYTE v19[72]; // [rsp+30h] [rbp-48h] BYREF

  *a1 = a1 + 2;
  a1[1] = 0;
  *((_BYTE *)a1 + 16) = 0;
  s[0] = v19;
  sub_CEB5A0((__int64 *)s, a3, a4);
  v6 = s[0];
  v7 = 2 * (a4 - (_QWORD)a3);
  v17 = &s[0][(unsigned __int64)s[1]];
  v8 = (const char *)sub_2207820(v7);
  while ( 1 )
  {
    v9 = sub_2254FB0(a2, v8, v6, v7);
    v10 = v9;
    if ( v7 <= v9 )
    {
      v7 = v9 + 1;
      j_j___libc_free_0_0((unsigned __int64)v8);
      v8 = (const char *)sub_2207820(v7);
      v10 = sub_2254FB0(a2, v8, v6, v7);
    }
    if ( v10 > 0x3FFFFFFFFFFFFFFFLL - a1[1] )
      sub_4262D8((__int64)"basic_string::append");
    sub_2241490(a1, v8);
    v11 = &v6[strlen(v6)];
    if ( v17 == v11 )
      break;
    v12 = a1[1];
    v6 = v11 + 1;
    v16 = v12 + 1;
    v13 = *a1;
    if ( a1 + 2 == (_QWORD *)*a1 )
      v14 = 15;
    else
      v14 = a1[2];
    if ( v16 > v14 )
    {
      sub_2240BB0(a1, v12, 0, 0, 1);
      v13 = *a1;
    }
    *(_BYTE *)(v13 + v12) = 0;
    a1[1] = v16;
    *(_BYTE *)(*a1 + v12 + 1) = 0;
  }
  j_j___libc_free_0_0((unsigned __int64)v8);
  if ( s[0] != v19 )
    j___libc_free_0((unsigned __int64)s[0]);
  return a1;
}
