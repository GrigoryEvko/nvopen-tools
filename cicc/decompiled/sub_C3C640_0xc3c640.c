// Function: sub_C3C640
// Address: 0xc3c640
//
_QWORD *__fastcall sub_C3C640(_QWORD *a1, __int64 a2, _QWORD *a3)
{
  _QWORD *v5; // rax
  __int64 v6; // rdi
  bool v7; // cc
  _QWORD *v8; // rbx
  __int64 *v9; // rax
  __int64 v10; // rax
  _DWORD *v11; // rax
  _DWORD *v12; // r15
  __int64 v13; // rax
  __int64 v14; // rdi
  _QWORD *v16; // [rsp+8h] [rbp-48h]
  __int64 v17; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v18; // [rsp+18h] [rbp-38h]

  *a1 = a2;
  v5 = (_QWORD *)sub_2207820(56);
  if ( v5 )
  {
    v6 = (__int64)(v5 + 1);
    v7 = *((_DWORD *)a3 + 2) <= 0x40u;
    *v5 = 2;
    v8 = v5;
    v16 = v5 + 1;
    v9 = a3;
    if ( !v7 )
      v9 = (__int64 *)*a3;
    v10 = *v9;
    v18 = 64;
    v17 = v10;
    v11 = sub_C33340();
    v12 = v11;
    if ( v11 == dword_3F657A0 )
      sub_C3C640(v6, v11, &v17);
    else
      sub_C3B160(v6, dword_3F657A0, &v17);
    if ( v18 > 0x40 && v17 )
      j_j___libc_free_0_0(v17);
    if ( *((_DWORD *)a3 + 2) > 0x40u )
      a3 = (_QWORD *)*a3;
    v13 = a3[1];
    v14 = (__int64)(v8 + 4);
    v18 = 64;
    v17 = v13;
    if ( v12 == dword_3F657A0 )
      sub_C3C640(v14, v12, &v17);
    else
      sub_C3B160(v14, dword_3F657A0, &v17);
    if ( v18 > 0x40 && v17 )
      j_j___libc_free_0_0(v17);
  }
  else
  {
    v16 = 0;
  }
  a1[1] = v16;
  return v16;
}
