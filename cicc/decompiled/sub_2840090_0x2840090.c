// Function: sub_2840090
// Address: 0x2840090
//
__int64 __fastcall sub_2840090(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r12
  unsigned int v5; // r13d
  unsigned __int16 v7; // ax
  unsigned int v8; // eax
  __int64 v9; // rax
  __int64 v10; // rdi
  _QWORD v11[10]; // [rsp+0h] [rbp-50h] BYREF

  v4 = *(_QWORD *)(a2 - 8);
  if ( *(_BYTE *)v4 == 61 )
  {
    v7 = *(_WORD *)(v4 + 2);
    if ( ((v7 >> 7) & 6) == 0 && (v7 & 1) == 0 )
    {
      LOBYTE(v8) = sub_D484B0(a1[5], *(_QWORD *)(a2 - 8), (*(_WORD *)(v4 + 2) >> 7) & 6, a4);
      v5 = v8;
      if ( (_BYTE)v8 )
      {
        v9 = *(_QWORD *)(v4 - 32);
        v10 = *a1;
        v11[1] = -1;
        v11[0] = v9;
        memset(&v11[2], 0, 32);
        if ( (sub_CF5020(v10, (__int64)v11, 0) & 2) == 0 )
          return v5;
        if ( (*(_BYTE *)(v4 + 7) & 0x20) != 0 )
        {
          LOBYTE(v5) = sub_B91C10(v4, 6) != 0;
          return v5;
        }
      }
    }
  }
  return 0;
}
