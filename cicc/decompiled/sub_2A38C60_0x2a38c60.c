// Function: sub_2A38C60
// Address: 0x2a38c60
//
void __fastcall sub_2A38C60(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rdx

  if ( a3 == 188 )
  {
    v5 = 1;
    v6 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    goto LABEL_11;
  }
  if ( a3 > 0xBC )
  {
    if ( a3 <= 0x168 )
    {
      if ( a3 > 0x165 )
        goto LABEL_6;
      return;
    }
    if ( a3 != 362 )
      return;
    goto LABEL_10;
  }
  if ( a3 == 124 )
  {
LABEL_10:
    v5 = 2;
    v6 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
LABEL_11:
    sub_2A38760(a1, *(_QWORD *)(a2 + 32 * (v5 - v6)), a4);
    goto LABEL_7;
  }
  if ( a3 <= 0x7C )
  {
    if ( a3 - 121 <= 2 )
      goto LABEL_6;
  }
  else if ( a3 == 187 )
  {
LABEL_6:
    sub_2A38760(a1, *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))), a4);
    sub_2A38830(a1, *(unsigned __int8 **)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))), 1, a4);
LABEL_7:
    sub_2A38830(a1, *(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), 0, a4);
  }
}
