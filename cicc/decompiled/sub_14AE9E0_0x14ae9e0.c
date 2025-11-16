// Function: sub_14AE9E0
// Address: 0x14ae9e0
//
__int64 __fastcall sub_14AE9E0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 i; // rbx

  v2 = **(_QWORD **)(a2 + 32);
  if ( *(_QWORD *)(a1 + 40) != v2 )
    return 0;
  for ( i = *(_QWORD *)(v2 + 48); !i; i = *(_QWORD *)(i + 8) )
  {
    if ( !(unsigned __int8)sub_14AE440(0) )
      return 0;
LABEL_6:
    ;
  }
  if ( a1 != i - 24 )
  {
    if ( !(unsigned __int8)sub_14AE440(i - 24) )
      return 0;
    goto LABEL_6;
  }
  return 1;
}
