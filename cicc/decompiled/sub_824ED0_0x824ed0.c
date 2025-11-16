// Function: sub_824ED0
// Address: 0x824ed0
//
__int64 __fastcall sub_824ED0(_QWORD *a1)
{
  __int64 v1; // rcx
  __int64 v2; // r12
  __int64 result; // rax

  v1 = a1[6];
  v2 = a1[5];
  if ( v1 )
    result = sub_685610(8u, (unsigned int)(v1 != 1) + 3232, *(_QWORD *)(a1[3] + 8LL), v1);
  if ( v2 )
    return sub_685610(5u, (unsigned int)(v2 != 1) + 3230, *(_QWORD *)(a1[3] + 8LL), v2);
  return result;
}
