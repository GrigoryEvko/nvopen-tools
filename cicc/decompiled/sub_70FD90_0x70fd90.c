// Function: sub_70FD90
// Address: 0x70fd90
//
__int64 __fastcall sub_70FD90(__int64 *a1, __int64 a2)
{
  __int64 result; // rax

  sub_724C70(a2, 12);
  sub_7249B0(a2, 1);
  *(_QWORD *)(a2 + 184) = a1;
  result = *a1;
  *(_QWORD *)(a2 + 128) = *a1;
  return result;
}
