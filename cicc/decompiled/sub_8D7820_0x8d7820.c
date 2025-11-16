// Function: sub_8D7820
// Address: 0x8d7820
//
__int64 __fastcall sub_8D7820(__int64 a1, __int64 a2, int a3, int a4)
{
  __int64 i; // rdx
  __int64 v7; // rax
  __int64 v8; // rcx
  unsigned int v9; // r8d
  int v10; // edx
  int v11; // eax

  for ( ; *(_BYTE *)(a1 + 140) == 12; a1 = *(_QWORD *)(a1 + 160) )
    ;
  for ( i = *(_QWORD *)(a1 + 168); *(_BYTE *)(a2 + 140) == 12; a2 = *(_QWORD *)(a2 + 160) )
    ;
  v7 = *(_QWORD *)(a2 + 168);
  v8 = *(_QWORD *)(v7 + 40);
  if ( !*(_QWORD *)(i + 40) )
    return v8 == 0;
  v9 = 0;
  if ( !v8 || ((*(_BYTE *)(v7 + 19) ^ *(_BYTE *)(i + 19)) & 0xC0) != 0 )
    return v9;
  v10 = *(_BYTE *)(i + 18) & 0x7F;
  v11 = *(_BYTE *)(v7 + 18) & 0x7F;
  if ( !qword_4D0495C )
    return v10 == v11;
  if ( !a4 && v10 != v11 )
  {
    if ( a3 )
      return (v10 & ~v11) == 0;
    return v9;
  }
  return 1;
}
