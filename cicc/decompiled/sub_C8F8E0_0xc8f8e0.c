// Function: sub_C8F8E0
// Address: 0xc8f8e0
//
__int64 __fastcall sub_C8F8E0(__int64 *a1, __int64 a2, __int64 a3, __int64 *a4)
{
  unsigned int v5; // r12d
  __int64 v8; // rax
  _QWORD *v9; // rdi
  _QWORD *v10; // rdi
  __int64 v11; // [rsp+0h] [rbp-60h] BYREF
  char v12; // [rsp+10h] [rbp-50h]
  __int64 v13[8]; // [rsp+20h] [rbp-40h] BYREF

  v5 = 0;
  sub_C8E8D0(&v11, (__int64)a1, a2, a4);
  if ( (v12 & 1) != 0 )
    return v5;
  v8 = v11;
  v9 = (_QWORD *)a1[1];
  v13[2] = a3;
  v11 = 0;
  v13[1] = 0;
  v13[0] = v8;
  if ( v9 == (_QWORD *)a1[2] )
  {
    sub_C12520(a1, (__int64)v9, (__int64)v13);
    v10 = (_QWORD *)a1[1];
  }
  else
  {
    if ( v9 )
    {
      sub_C8EDF0(v9, v13);
      v9 = (_QWORD *)a1[1];
    }
    v10 = v9 + 3;
    a1[1] = (__int64)v10;
  }
  v5 = -1431655765 * (((__int64)v10 - *a1) >> 3);
  sub_C8EE20(v13);
  if ( (v12 & 1) != 0 || !v11 )
    return v5;
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v11 + 8LL))(v11);
  return v5;
}
