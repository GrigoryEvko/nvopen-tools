// Function: sub_ACD6D0
// Address: 0xacd6d0
//
__int64 __fastcall sub_ACD6D0(__int64 *a1)
{
  __int64 v1; // rbx
  __int64 result; // rax
  __int64 v3; // rax

  v1 = *a1;
  result = *(_QWORD *)(*a1 + 2184);
  if ( !result )
  {
    v3 = sub_BCB2A0(a1);
    result = sub_ACD640(v3, 1, 0);
    *(_QWORD *)(v1 + 2184) = result;
  }
  return result;
}
