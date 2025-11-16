// Function: sub_25A1010
// Address: 0x25a1010
//
__int64 __fastcall sub_25A1010(__int64 a1)
{
  __int64 v1; // r12
  __int64 result; // rax
  unsigned __int64 v3[3]; // [rsp+0h] [rbp-20h] BYREF

  v1 = *(_QWORD *)(a1 + 8);
  sub_250D230(v3, **(_QWORD **)a1, 4, 0);
  result = sub_251C7D0(v1, v3[0], v3[1], 0, 2, 0, 1);
  if ( result )
    result += 104;
  return result;
}
