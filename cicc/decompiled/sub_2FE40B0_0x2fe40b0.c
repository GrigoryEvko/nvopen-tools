// Function: sub_2FE40B0
// Address: 0x2fe40b0
//
__int64 __fastcall sub_2FE40B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  unsigned __int16 *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax

  v5 = *(unsigned __int16 **)(a3 + 48);
  v6 = 1;
  v7 = *v5;
  if ( (_WORD)v7 != 1 )
  {
    a5 = 0;
    if ( !(_WORD)v7 )
      return 0;
    v6 = (unsigned __int16)v7;
    if ( !*(_QWORD *)(a1 + 8 * v7 + 112) )
      return 0;
  }
  LOBYTE(a5) = *(_BYTE *)(a1 + 500 * v6 + 6565) == 0;
  return a5;
}
