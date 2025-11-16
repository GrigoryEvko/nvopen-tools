// Function: sub_11DB340
// Address: 0x11db340
//
__int64 __fastcall sub_11DB340(_DWORD **a1, double a2)
{
  _DWORD *v2; // r14
  unsigned int v3; // r13d
  __int64 v4; // r14
  void *v5; // rax
  void *v6; // rbx
  unsigned int v8; // eax
  _QWORD *i; // rbx
  unsigned int v10; // eax
  __int64 v11; // [rsp+8h] [rbp-78h]
  void *v12; // [rsp+10h] [rbp-70h] BYREF
  _QWORD *v13; // [rsp+18h] [rbp-68h]
  __int64 v14[10]; // [rsp+30h] [rbp-50h] BYREF

  v2 = sub_C33320();
  sub_C3B1B0((__int64)v14, a2);
  sub_C407B0(&v12, v14, v2);
  sub_C338F0((__int64)v14);
  v3 = 0;
  sub_C41640((__int64 *)&v12, *a1, 1, (bool *)v14);
  v4 = (__int64)v12;
  v11 = (__int64)*a1;
  v5 = sub_C33340();
  v6 = v5;
  if ( v11 == v4 )
  {
    if ( (void *)v4 != v5 )
    {
      LOBYTE(v8) = sub_C33D00((__int64)a1, (__int64)&v12);
      v3 = v8;
      if ( v12 != v6 )
        goto LABEL_3;
      goto LABEL_7;
    }
    LOBYTE(v10) = sub_C3E590((__int64)a1, (__int64)&v12);
    v4 = (__int64)v12;
    v3 = v10;
  }
  if ( (void *)v4 != v6 )
  {
LABEL_3:
    sub_C338F0((__int64)&v12);
    return v3;
  }
LABEL_7:
  if ( v13 )
  {
    for ( i = &v13[3 * *(v13 - 1)]; v13 != i; sub_91D830(i) )
      i -= 3;
    j_j_j___libc_free_0_0(i - 1);
  }
  return v3;
}
