// Function: sub_1815C10
// Address: 0x1815c10
//
__int64 sub_1815C10()
{
  __int64 v0; // rax
  __int64 v1; // r13
  _QWORD *v2; // rbx
  _QWORD *v3; // r12
  _QWORD *v5; // [rsp+0h] [rbp-40h] BYREF
  _QWORD *v6; // [rsp+8h] [rbp-38h]
  __int64 v7; // [rsp+10h] [rbp-30h]

  v5 = 0;
  v6 = 0;
  v7 = 0;
  v0 = sub_22077B0(536);
  v1 = v0;
  if ( v0 )
    sub_18156A0(v0, (__int64 *)&v5, 0, 0);
  v2 = v6;
  v3 = v5;
  if ( v6 != v5 )
  {
    do
    {
      if ( (_QWORD *)*v3 != v3 + 2 )
        j_j___libc_free_0(*v3, v3[2] + 1LL);
      v3 += 4;
    }
    while ( v2 != v3 );
    v3 = v5;
  }
  if ( v3 )
    j_j___libc_free_0(v3, v7 - (_QWORD)v3);
  return v1;
}
