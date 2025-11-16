// Function: sub_832ED0
// Address: 0x832ed0
//
__int64 __fastcall sub_832ED0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // r14
  __int64 v4; // rbx
  _BOOL4 v5; // eax
  _BYTE v7[8]; // [rsp+0h] [rbp-50h] BYREF
  int v8; // [rsp+8h] [rbp-48h]

  for ( ; *(_BYTE *)(a2 + 140) == 12; a2 = *(_QWORD *)(a2 + 160) )
    ;
  v2 = *(_QWORD *)(a2 + 160);
  v3 = *(_QWORD *)a1;
  if ( !v2 )
    return 0;
  while ( 1 )
  {
    v4 = *(_QWORD *)(v2 + 120);
    if ( v3 == v4 )
      break;
    if ( (unsigned int)sub_8DED30(v3, *(_QWORD *)(v2 + 120), 3) )
      break;
    if ( (unsigned int)sub_8D2E30(v4) )
    {
      v5 = sub_6EB660(a1);
      if ( (unsigned int)sub_8DFA20(
                           v3,
                           *(_BYTE *)(a1 + 16) == 2,
                           (*(_BYTE *)(a1 + 19) & 0x10) != 0,
                           v5,
                           (int)a1 + 144,
                           v4,
                           0,
                           0,
                           171,
                           (__int64)v7,
                           0) )
      {
        if ( !v8 )
          break;
      }
    }
    v2 = *(_QWORD *)(v2 + 112);
    if ( !v2 )
      return 0;
  }
  return v2;
}
