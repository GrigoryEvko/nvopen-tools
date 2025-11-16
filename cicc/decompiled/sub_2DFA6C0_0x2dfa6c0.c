// Function: sub_2DFA6C0
// Address: 0x2dfa6c0
//
__int64 __fastcall sub_2DFA6C0(_QWORD *a1)
{
  __int64 *v1; // r13

  v1 = (__int64 *)a1[25];
  *a1 = &unk_4A28318;
  if ( v1 )
  {
    sub_2DFA680(v1);
    j_j___libc_free_0((unsigned __int64)v1);
  }
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
