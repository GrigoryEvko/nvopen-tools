// Function: sub_C40510
// Address: 0xc40510
//
__int64 __fastcall sub_C40510(_QWORD **a1)
{
  __int64 *v1; // r13
  char v3; // al
  void **v4; // rbx
  void *v5; // r12
  __int64 v6; // [rsp+0h] [rbp-30h] BYREF
  void **v7; // [rsp+8h] [rbp-28h]

  LODWORD(v1) = 0;
  if ( (unsigned int)sub_C3CE50((__int64)a1) != 2 )
    return (unsigned int)v1;
  v1 = &v6;
  sub_C3C790(&v6, a1);
  v3 = sub_C3CE80((__int64)a1);
  sub_C3CF90((__int64)&v6, v3);
  LOBYTE(v1) = (unsigned int)sub_C3E510((__int64)&v6, (__int64)a1) == 1;
  if ( !v7 )
    return (unsigned int)v1;
  v4 = &v7[3 * (_QWORD)*(v7 - 1)];
  if ( v7 != v4 )
  {
    v5 = sub_C33340();
    do
    {
      while ( 1 )
      {
        v4 -= 3;
        if ( *v4 == v5 )
          break;
        sub_C338F0((__int64)v4);
        if ( v7 == v4 )
          goto LABEL_9;
      }
      sub_969EE0((__int64)v4);
    }
    while ( v7 != v4 );
  }
LABEL_9:
  j_j_j___libc_free_0_0(v4 - 1);
  return (unsigned int)v1;
}
