// Function: sub_1DF8200
// Address: 0x1df8200
//
__int64 __fastcall sub_1DF8200(__int64 a1, int a2)
{
  __int64 result; // rax

  if ( a2 < 0 )
    result = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL * (a2 & 0x7FFFFFFF) + 8);
  else
    result = *(_QWORD *)(*(_QWORD *)(a1 + 272) + 8LL * (unsigned int)a2);
  while ( result && ((*(_BYTE *)(result + 3) & 0x10) != 0 || (*(_BYTE *)(result + 4) & 8) != 0) )
    result = *(_QWORD *)(result + 32);
  return result;
}
