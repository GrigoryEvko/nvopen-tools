// Function: sub_2A109A0
// Address: 0x2a109a0
//
unsigned __int64 __fastcall sub_2A109A0(__int64 *a1)
{
  unsigned __int64 *v1; // r12
  unsigned __int64 result; // rax
  __int64 v3; // rax
  __int64 v4; // r13
  unsigned __int64 v5; // r14

  v1 = (unsigned __int64 *)a1[3];
  result = *v1;
  if ( !*v1 )
  {
    v3 = sub_22077B0(0x168u);
    v4 = v3;
    if ( v3 )
      sub_1043250(v3, *a1, a1[1], a1[2]);
    v5 = *v1;
    *v1 = v4;
    if ( v5 )
    {
      sub_103C970(v5);
      j_j___libc_free_0(v5);
    }
    return *(_QWORD *)a1[3];
  }
  return result;
}
