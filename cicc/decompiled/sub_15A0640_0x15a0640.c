// Function: sub_15A0640
// Address: 0x15a0640
//
__int64 __fastcall sub_15A0640(__int64 a1)
{
  __int64 result; // rax

  result = sub_159C540(*(__int64 **)a1);
  if ( *(_BYTE *)(a1 + 8) == 16 )
    return sub_15A0390(*(_QWORD *)(a1 + 32), result);
  return result;
}
