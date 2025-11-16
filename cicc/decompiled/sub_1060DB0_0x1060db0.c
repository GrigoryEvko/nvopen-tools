// Function: sub_1060DB0
// Address: 0x1060db0
//
__int64 __fastcall sub_1060DB0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rax

  result = *(_QWORD *)(a1 + 24);
  if ( !result )
  {
    v2 = sub_BCB2D0(*(_QWORD **)(*(_QWORD *)a1 + 72LL));
    return sub_ACD640(v2, *(unsigned int *)(a1 + 32), 0);
  }
  return result;
}
