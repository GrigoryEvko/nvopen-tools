// Function: sub_2FF62B0
// Address: 0x2ff62b0
//
__int64 __fastcall sub_2FF62B0(__int64 a1, _QWORD *a2, unsigned int a3)
{
  unsigned int v5; // esi
  __int64 v6; // rdx
  __int64 result; // rax
  unsigned __int64 v8; // rcx
  __int64 v9; // rdx

  v5 = a3;
  v6 = *(unsigned int *)(*(_QWORD *)(a1 + 8) + 24LL * a3 + 8);
  result = *(_QWORD *)(a1 + 56);
  v8 = v5;
  v9 = result + 2 * v6;
  if ( v9 )
  {
    while ( 1 )
    {
      v9 += 2;
      *(_QWORD *)(*a2 + ((v8 >> 3) & 0x1FF8)) |= 1LL << v8;
      result = (unsigned int)*(__int16 *)(v9 - 2);
      if ( !*(_WORD *)(v9 - 2) )
        break;
      v5 += result;
      v8 = v5;
    }
  }
  return result;
}
