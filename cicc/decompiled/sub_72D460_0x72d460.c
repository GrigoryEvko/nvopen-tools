// Function: sub_72D460
// Address: 0x72d460
//
__int64 __fastcall sub_72D460(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 result; // rax

  sub_724C70(a2, 6);
  v2 = *(_BYTE *)(a1 + 173);
  *(_QWORD *)(a2 + 184) = a1;
  if ( v2 != 2 )
    v2 = 3;
  *(_BYTE *)(a2 + 176) = v2;
  result = sub_72D2E0(*(_QWORD **)(a1 + 128));
  *(_QWORD *)(a2 + 128) = result;
  return result;
}
