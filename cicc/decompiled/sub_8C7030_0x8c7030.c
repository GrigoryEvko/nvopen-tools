// Function: sub_8C7030
// Address: 0x8c7030
//
__int64 __fastcall sub_8C7030(__int64 a1)
{
  __int64 result; // rax

  for ( result = *(_QWORD *)(a1 + 32);
        (*(_BYTE *)(result + 89) & 4) != 0
     && ((unsigned __int8)(*(_BYTE *)(result + 140) - 9) > 2u
      || !*(_QWORD *)(result + 8)
      || *(char *)(result + 177) >= 0
      || !*(_QWORD *)(*(_QWORD *)(result + 168) + 168LL))
     && !*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(result + 40) + 32LL) + 32LL);
        result = *(_QWORD *)(*(_QWORD *)(result + 40) + 32LL) )
  {
    ;
  }
  return result;
}
