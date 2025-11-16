// Function: sub_33CA5E0
// Address: 0x33ca5e0
//
_BOOL8 __fastcall sub_33CA5E0(__int64 a1, __int64 a2, __int64 *a3)
{
  _QWORD *v4; // rbx
  _DWORD *v5; // rax
  _BOOL4 v6; // r12d
  _QWORD *i; // rbx
  _QWORD v9[3]; // [rsp+0h] [rbp-60h] BYREF
  bool v10; // [rsp+1Fh] [rbp-41h] BYREF
  _QWORD *v11; // [rsp+20h] [rbp-40h] BYREF
  _QWORD *v12; // [rsp+28h] [rbp-38h]

  v9[0] = a1;
  v9[1] = a2;
  v4 = sub_C33340();
  if ( (_QWORD *)*a3 == v4 )
    sub_C3C790(&v11, (_QWORD **)a3);
  else
    sub_C33EB0(&v11, a3);
  v5 = sub_300AC80((unsigned __int16 *)v9, (__int64)a3);
  sub_C41640((__int64 *)&v11, v5, 1, &v10);
  v6 = !v10;
  if ( v11 != v4 )
  {
    sub_C338F0((__int64)&v11);
    return v6;
  }
  if ( !v12 )
    return v6;
  for ( i = &v12[3 * *(v12 - 1)]; v12 != i; sub_91D830(i) )
    i -= 3;
  j_j_j___libc_free_0_0((unsigned __int64)(i - 1));
  return v6;
}
