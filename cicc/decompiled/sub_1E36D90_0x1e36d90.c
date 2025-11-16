// Function: sub_1E36D90
// Address: 0x1e36d90
//
unsigned __int64 __fastcall sub_1E36D90(__int64 **a1, __int64 a2)
{
  unsigned __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // r13
  unsigned __int64 result; // rax

  sub_1E36D20((__int64)a1, a2);
  v2 = 0;
  v3 = sub_15E0530(**a1);
  v4 = v3;
  if ( *(_BYTE *)(a2 + 80) )
    v2 = *(_QWORD *)(a2 + 72);
  result = sub_1602780(v3);
  if ( result <= v2 )
    return sub_16027F0(v4, a2);
  return result;
}
