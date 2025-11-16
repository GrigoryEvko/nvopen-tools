// Function: sub_989010
// Address: 0x989010
//
__int64 __fastcall sub_989010(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v5; // rsi
  __int64 v6; // rax
  unsigned int v7; // [rsp-2Ch] [rbp-2Ch] BYREF

  v2 = *(_QWORD *)(a1 - 32);
  if ( !v2 || *(_BYTE *)v2 )
    return 0;
  if ( *(_QWORD *)(v2 + 24) == *(_QWORD *)(a1 + 80) )
  {
    if ( (*(_BYTE *)(v2 + 33) & 0x20) != 0 )
      return *(unsigned int *)(v2 + 36);
    if ( (*(_BYTE *)(v2 + 32) & 0xFu) - 7 > 1
      && a2
      && (!(unsigned __int8)sub_A73ED0(a1 + 72, 23) && !(unsigned __int8)sub_B49560(a1, 23)
       || (unsigned __int8)sub_A73ED0(a1 + 72, 4)
       || (unsigned __int8)sub_B49560(a1, 4)) )
    {
      v5 = *(_QWORD *)(a1 - 32);
      if ( v5 )
      {
        if ( !*(_BYTE *)v5 && *(_QWORD *)(v5 + 24) == *(_QWORD *)(a1 + 80) && sub_981210(*a2, v5, &v7) )
        {
          if ( (unsigned __int8)sub_B49E20(a1) )
          {
            v6 = v7 - 160;
            if ( (unsigned int)v6 <= 0x156 )
              return word_3F1F540[v6];
          }
        }
      }
    }
  }
  return 0;
}
