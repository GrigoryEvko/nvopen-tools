// Function: sub_30CAC80
// Address: 0x30cac80
//
__int64 __fastcall sub_30CAC80(_QWORD *a1)
{
  __int64 result; // rax

  result = a1[1];
  if ( *(_QWORD *)(result + 72) )
    return sub_36FCF50(*(_QWORD *)(result + 72), a1[2], a1[3]);
  return result;
}
