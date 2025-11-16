// Function: sub_622E70
// Address: 0x622e70
//
__int64 __fastcall sub_622E70(__int64 *a1)
{
  __int64 v1; // rdx
  __int64 result; // rax

  v1 = *a1;
  if ( !*a1 )
    return sub_6851C0(668, a1 + 6);
  if ( (unsigned __int8)(*(_BYTE *)(v1 + 80) - 10) > 1u )
    return sub_6851C0(668, a1 + 6);
  result = sub_736C60(20, *(_QWORD *)(*(_QWORD *)(v1 + 88) + 104LL));
  if ( !result )
    return sub_6851C0(668, a1 + 6);
  return result;
}
