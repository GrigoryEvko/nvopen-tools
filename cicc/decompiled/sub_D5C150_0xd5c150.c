// Function: sub_D5C150
// Address: 0xd5c150
//
__int64 __fastcall sub_D5C150(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v6; // rsi
  unsigned int v7; // eax
  unsigned int v8; // eax

  if ( !*(_BYTE *)(a2 + 16) || !*(_BYTE *)(a3 + 16) )
  {
    *(_BYTE *)(a1 + 16) = 0;
    return a1;
  }
  if ( a4 == 3 )
  {
    v6 = a3;
    if ( (int)sub_C4C880(a2, a3) >= 0 )
      v6 = a2;
    v8 = *(_DWORD *)(v6 + 8);
    *(_DWORD *)(a1 + 8) = v8;
    if ( v8 <= 0x40 )
      goto LABEL_8;
LABEL_13:
    sub_C43780(a1, (const void **)v6);
    goto LABEL_9;
  }
  v6 = a3;
  if ( (int)sub_C4C880(a2, a3) <= 0 )
    v6 = a2;
  v7 = *(_DWORD *)(v6 + 8);
  *(_DWORD *)(a1 + 8) = v7;
  if ( v7 > 0x40 )
    goto LABEL_13;
LABEL_8:
  *(_QWORD *)a1 = *(_QWORD *)v6;
LABEL_9:
  *(_BYTE *)(a1 + 16) = 1;
  return a1;
}
