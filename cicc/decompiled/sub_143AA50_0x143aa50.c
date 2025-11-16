// Function: sub_143AA50
// Address: 0x143aa50
//
unsigned __int64 __fastcall sub_143AA50(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v2; // r13
  __int64 v3; // rax
  unsigned __int64 result; // rax
  __int64 v5; // rax

  v2 = 0;
  sub_143A9E0((__int64)a1, a2);
  if ( *(_BYTE *)(a2 + 80) )
    v2 = *(_QWORD *)(a2 + 72);
  v3 = sub_15E0530(*a1);
  result = sub_1602780(v3);
  if ( result <= v2 )
  {
    v5 = sub_15E0530(*a1);
    return sub_16027F0(v5, a2);
  }
  return result;
}
