// Function: sub_E8A1F0
// Address: 0xe8a1f0
//
__int64 __fastcall sub_E8A1F0(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(*(_QWORD *)(a1 + 296) + 24LL);
  *(_BYTE *)(result + 80) = 1;
  return result;
}
