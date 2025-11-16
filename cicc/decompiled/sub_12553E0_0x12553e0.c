// Function: sub_12553E0
// Address: 0x12553e0
//
__int64 __fastcall sub_12553E0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 result; // rax

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 40) = 0;
  v2 = *a2;
  *(_QWORD *)(a1 + 16) = a2;
  *(_QWORD *)(a1 + 24) = 0;
  result = (*(__int64 (__fastcall **)(__int64 *))(v2 + 48))(a2);
  if ( (result & 2) == 0 )
  {
    result = (*(__int64 (__fastcall **)(__int64 *))(*a2 + 40))(a2);
    *(_BYTE *)(a1 + 40) = 1;
    *(_QWORD *)(a1 + 32) = result;
  }
  return result;
}
