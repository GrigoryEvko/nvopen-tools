// Function: sub_E99830
// Address: 0xe99830
//
__int64 __fastcall sub_E99830(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // rbx
  __int64 (*v4)(); // rdx

  result = sub_E99590(a1, a2);
  if ( result )
  {
    v3 = result;
    v4 = *(__int64 (**)())(*(_QWORD *)a1 + 88LL);
    result = 1;
    if ( v4 != sub_E97650 )
      result = ((__int64 (__fastcall *)(__int64))v4)(a1);
    *(_QWORD *)(v3 + 40) = result;
  }
  return result;
}
