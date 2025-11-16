// Function: sub_72D410
// Address: 0x72d410
//
__int64 __fastcall sub_72D410(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  sub_724C70(a2, 6);
  *(_QWORD *)(a2 + 184) = a1;
  *(_BYTE *)(a2 + 176) = 2;
  result = sub_72D2E0(*(_QWORD **)(a1 + 128));
  *(_QWORD *)(a2 + 128) = result;
  return result;
}
