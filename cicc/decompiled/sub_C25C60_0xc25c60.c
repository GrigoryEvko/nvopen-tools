// Function: sub_C25C60
// Address: 0xc25c60
//
__int64 __fastcall sub_C25C60(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r12
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v6; // [rsp+0h] [rbp-40h]
  __int64 v7; // [rsp+8h] [rbp-38h]
  __int64 v8; // [rsp+10h] [rbp-30h] BYREF
  __int64 v9[5]; // [rsp+18h] [rbp-28h] BYREF

  v1 = (_QWORD *)a1[34];
  v2 = (_QWORD *)a1[33];
  *a1 = &unk_49DBD38;
  if ( v1 != v2 )
  {
    do
    {
      if ( (_QWORD *)*v2 != v2 + 2 )
        j_j___libc_free_0(*v2, v2[2] + 1LL);
      v2 += 4;
    }
    while ( v1 != v2 );
    v2 = (_QWORD *)a1[33];
  }
  if ( v2 )
    j_j___libc_free_0(v2, a1[35] - (_QWORD)v2);
  v3 = a1[30];
  a1[30] = 0;
  v6 = 0;
  v7 = 0;
  v9[0] = v3 | 1;
  sub_C25920(&v8, v9);
  if ( (v8 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    BUG();
  if ( (v9[0] & 1) != 0 || (v9[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(v9);
  v4 = a1[30];
  if ( (v4 & 1) != 0 || (v4 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(a1 + 30);
  sub_C201C0((__int64)a1);
  return j_j___libc_free_0(a1, 288);
}
