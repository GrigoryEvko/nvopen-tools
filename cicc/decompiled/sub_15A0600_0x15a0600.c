// Function: sub_15A0600
// Address: 0x15a0600
//
__int64 __fastcall sub_15A0600(__int64 a1)
{
  __int64 result; // rax

  result = sub_159C4F0(*(__int64 **)a1);
  if ( *(_BYTE *)(a1 + 8) == 16 )
    return sub_15A0390(*(_QWORD *)(a1 + 32), result);
  return result;
}
