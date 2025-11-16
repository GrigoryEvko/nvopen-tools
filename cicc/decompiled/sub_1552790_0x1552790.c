// Function: sub_1552790
// Address: 0x1552790
//
_BYTE *__fastcall sub_1552790(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v5; // r14
  _BYTE *v6; // rax
  __int64 v7; // rdi
  _BYTE *v8; // rax
  __int64 v10; // [rsp+8h] [rbp-48h] BYREF
  const char *v11[2]; // [rsp+10h] [rbp-40h] BYREF
  __int64 v12; // [rsp+20h] [rbp-30h] BYREF

  v10 = a3;
  if ( !a2 )
    return (_BYTE *)sub_1263B40(*a1, "<null operand!>");
  v3 = (__int64)(a1 + 5);
  sub_154DAA0((__int64)(a1 + 5), *a2, *a1);
  if ( v10 )
  {
    v5 = *a1;
    v6 = *(_BYTE **)(*a1 + 24);
    if ( (unsigned __int64)v6 >= *(_QWORD *)(*a1 + 16) )
    {
      v5 = sub_16E7DE0(*a1, 32);
    }
    else
    {
      *(_QWORD *)(v5 + 24) = v6 + 1;
      *v6 = 32;
    }
    sub_155F820(v11, &v10, 0);
    sub_16E7EE0(v5, v11[0], v11[1]);
    if ( (__int64 *)v11[0] != &v12 )
      j_j___libc_free_0(v11[0], v12 + 1);
  }
  v7 = *a1;
  v8 = *(_BYTE **)(*a1 + 24);
  if ( (unsigned __int64)v8 >= *(_QWORD *)(*a1 + 16) )
  {
    sub_16E7DE0(v7, 32);
  }
  else
  {
    *(_QWORD *)(v7 + 24) = v8 + 1;
    *v8 = 32;
  }
  return sub_1550E20(*a1, (__int64)a2, v3, a1[4], a1[1]);
}
