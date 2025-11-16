// Function: sub_15A3C50
// Address: 0x15a3c50
//
unsigned __int64 __fastcall sub_15A3C50(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  unsigned __int64 result; // rax

  v2 = a1;
  if ( *(_BYTE *)(a1 + 8) == 16 )
    v2 = **(_QWORD **)(a1 + 16);
  result = sub_159C0E0(*(__int64 **)a1, a2);
  if ( *(_BYTE *)(v2 + 8) == 15 )
    result = sub_15A3BA0(result, (__int64 **)v2, 0);
  if ( *(_BYTE *)(a1 + 8) == 16 )
    return sub_15A0390(*(_QWORD *)(a1 + 32), result);
  return result;
}
