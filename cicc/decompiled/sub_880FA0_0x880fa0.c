// Function: sub_880FA0
// Address: 0x880fa0
//
__int64 __fastcall sub_880FA0(__int64 a1)
{
  __int64 result; // rax

  result = 0;
  if ( dword_4F07588 )
  {
    while ( *(_BYTE *)(a1 + 140) == 12 )
      a1 = *(_QWORD *)(a1 + 160);
    return *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 96LL) + 72LL);
  }
  return result;
}
