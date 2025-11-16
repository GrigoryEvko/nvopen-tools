// Function: sub_DC2CB0
// Address: 0xdc2cb0
//
_QWORD *__fastcall sub_DC2CB0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rbx

  v4 = sub_D95540(a2);
  v5 = sub_D97050(a1, v4);
  if ( v5 == sub_D97050(a1, a3) )
    return (_QWORD *)a2;
  else
    return sub_DC2B70(a1, a2, a3, 0);
}
