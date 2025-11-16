// Function: sub_15A1070
// Address: 0x15a1070
//
__int64 __fastcall sub_15A1070(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = sub_159C0E0(*(__int64 **)a1, a2);
  if ( *(_BYTE *)(a1 + 8) == 16 )
    return sub_15A0390(*(_QWORD *)(a1 + 32), result);
  return result;
}
