// Function: sub_71ADE0
// Address: 0x71ade0
//
__int64 __fastcall sub_71ADE0(_QWORD *a1, __int64 *a2)
{
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 result; // rax

  v2 = *(_QWORD *)(qword_4F04C50 + 40LL);
  if ( *(_QWORD *)(v2 + 112) )
  {
    v3 = sub_71AD70(*(_QWORD *)(qword_4F04C50 + 40LL));
    *a1 = sub_73E1B0(v3, a2);
    v2 = *(_QWORD *)(v2 + 112);
  }
  else
  {
    *a1 = sub_73E870();
  }
  v4 = sub_71AD70(v2);
  result = sub_73E1B0(v4, a2);
  *a2 = result;
  return result;
}
