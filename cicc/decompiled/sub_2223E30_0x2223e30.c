// Function: sub_2223E30
// Address: 0x2223e30
//
_QWORD *__fastcall sub_2223E30(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  const wchar_t *v8; // rbx
  __int64 v9; // rbp
  unsigned __int64 v10; // r13
  unsigned __int64 v11; // rbp
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rdx
  const wchar_t *v14; // rbx
  __int64 v15; // r14
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  unsigned __int64 v19; // [rsp+0h] [rbp-78h]
  wchar_t *v20; // [rsp+8h] [rbp-70h]
  wchar_t *s[2]; // [rsp+20h] [rbp-58h] BYREF
  _BYTE v22[72]; // [rsp+30h] [rbp-48h] BYREF

  *a1 = a1 + 2;
  a1[1] = 0;
  *((_DWORD *)a1 + 4) = 0;
  s[0] = (wchar_t *)v22;
  sub_22520E0(s);
  v8 = s[0];
  v9 = a4 - a3;
  v10 = v9 >> 1;
  v20 = &s[0][(__int64)s[1]];
  if ( (unsigned __int64)v9 > 0x3FFFFFFFFFFFFFFCLL )
    sub_426640(s, a3);
  v11 = sub_2207820(4 * v10);
  while ( 1 )
  {
    v12 = sub_2255000(a2, v11, v8, v10);
    v13 = v12;
    if ( v10 <= v12 )
    {
      v10 = v12 + 1;
      j_j___libc_free_0_0(v11);
      if ( v10 > 0x1FFFFFFFFFFFFFFELL )
        sub_426640(v11, v11);
      v11 = sub_2207820(4 * v10);
      v13 = sub_2255000(a2, v11, v8, v10);
    }
    if ( v13 > 0xFFFFFFFFFFFFFFFLL - a1[1] )
      sub_4262D8((__int64)"basic_string::append");
    sub_2251F20(a1, v11);
    v14 = &v8[wcslen(v8)];
    if ( v20 == v14 )
      break;
    v15 = a1[1];
    v8 = v14 + 1;
    v19 = v15 + 1;
    v16 = *a1;
    if ( a1 + 2 == (_QWORD *)*a1 )
      v17 = 3;
    else
      v17 = a1[2];
    if ( v19 > v17 )
    {
      sub_2251880(a1, v15, 0, 0, 1);
      v16 = *a1;
    }
    *(_DWORD *)(v16 + 4 * v15) = 0;
    a1[1] = v19;
    *(_DWORD *)(v16 + 4 * v15 + 4) = 0;
  }
  j_j___libc_free_0_0(v11);
  if ( (_BYTE *)s[0] != v22 )
    j___libc_free_0((unsigned __int64)s[0]);
  return a1;
}
