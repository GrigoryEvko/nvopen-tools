// Function: sub_72D3B0
// Address: 0x72d3b0
//
__int64 __fastcall sub_72D3B0(__int64 a1, __int64 a2, int a3)
{
  __int64 result; // rax

  sub_724C70(a2, 6);
  *(_BYTE *)(a2 + 176) = 0;
  *(_QWORD *)(a2 + 184) = a1;
  result = sub_72D2E0(*(_QWORD **)(a1 + 152));
  *(_QWORD *)(a2 + 128) = result;
  if ( a3 )
    *(_BYTE *)(a1 + 192) |= 1u;
  return result;
}
