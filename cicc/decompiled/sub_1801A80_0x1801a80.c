// Function: sub_1801A80
// Address: 0x1801a80
//
_QWORD *__fastcall sub_1801A80(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        char *a7,
        __int64 a8,
        __int64 a9)
{
  _QWORD *v10; // rax
  _QWORD *v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rax
  _QWORD *v14; // rdi
  __int64 v15; // rax
  __int64 **v16; // rax
  __int64 v17; // rcx
  __int64 *v18; // rax
  __int64 v19; // r14
  __int64 v20; // rbx
  _QWORD *v21; // r12
  __int64 v23[4]; // [rsp+0h] [rbp-60h] BYREF
  const char *v24; // [rsp+20h] [rbp-40h] BYREF
  char v25; // [rsp+30h] [rbp-30h]
  char v26; // [rsp+31h] [rbp-2Fh]

  v10 = sub_1801990(a1, a7, a8, 1);
  v11 = (_QWORD *)*a1;
  v23[0] = (__int64)v10;
  v12 = sub_1643350(v11);
  v13 = sub_159C470(v12, (int)a9, 0);
  v14 = (_QWORD *)*a1;
  v23[1] = v13;
  v15 = sub_1643350(v14);
  v23[2] = sub_159C470(v15, SHIDWORD(a9), 0);
  v16 = (__int64 **)sub_15943F0(v23, 3, 0);
  v18 = (__int64 *)sub_159F090(v16, v23, 3, v17);
  v19 = *v18;
  v20 = (__int64)v18;
  v26 = 1;
  v24 = "___asan_gen_";
  v25 = 3;
  v21 = sub_1648A60(88, 1u);
  if ( v21 )
    sub_15E51E0((__int64)v21, (__int64)a1, v19, 1, 8, v20, (__int64)&v24, 0, 0, 0, 0);
  *((_BYTE *)v21 + 32) = v21[4] & 0x3F | 0x80;
  return v21;
}
