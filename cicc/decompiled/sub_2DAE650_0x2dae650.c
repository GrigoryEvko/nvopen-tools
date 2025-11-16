// Function: sub_2DAE650
// Address: 0x2dae650
//
__int64 __fastcall sub_2DAE650(__int64 *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rbx
  char v8; // al
  __int64 v9; // r14
  __int64 v11; // r15
  __int64 v12; // rcx
  unsigned __int16 v13; // r12
  _QWORD *v14; // r12
  int v15; // eax
  unsigned __int16 v16; // cx
  __int64 v17; // [rsp+8h] [rbp-38h]

  v6 = *a1;
  if ( (a2 & 0x80000000) != 0 )
  {
    v7 = *(_QWORD *)(*(_QWORD *)(v6 + 56) + 16LL * (a2 & 0x7FFFFFFF) + 8);
  }
  else
  {
    v6 = *(_QWORD *)(v6 + 304);
    v7 = *(_QWORD *)(v6 + 8LL * a2);
  }
  while ( 1 )
  {
    if ( !v7 )
      return 0;
    if ( (*(_BYTE *)(v7 + 3) & 0x10) == 0 )
    {
      v8 = *(_BYTE *)(v7 + 4);
      if ( (v8 & 8) == 0 )
        break;
    }
    v7 = *(_QWORD *)(v7 + 32);
  }
  v11 = 0;
  v9 = 0;
LABEL_10:
  if ( (v8 & 1) != 0
    || (v8 & 2) != 0
    || (*(_BYTE *)(v7 + 3) & 0x10) != 0 && (*(_DWORD *)v7 & 0xFFF00) == 0
    || (v12 = *(unsigned __int16 *)(*(_QWORD *)(v7 + 16) + 68LL), (_WORD)v12 == 7) )
  {
LABEL_19:
    while ( 1 )
    {
      v7 = *(_QWORD *)(v7 + 32);
      if ( !v7 )
        return v9;
      if ( (*(_BYTE *)(v7 + 3) & 0x10) == 0 )
      {
        v8 = *(_BYTE *)(v7 + 4);
        if ( (v8 & 8) == 0 )
          goto LABEL_10;
      }
    }
  }
  v13 = (*(_DWORD *)v7 >> 8) & 0xFFF;
  if ( (unsigned __int16)v12 > 0x14u
    || ((1LL << v12) & 0x180301) == 0
    || (v17 = *(_QWORD *)(v7 + 16), sub_2E88FE0(v17), v15 = *(_DWORD *)(*(_QWORD *)(v17 + 32) + 8LL), v15 >= 0) )
  {
    if ( !v13 )
      return sub_2EBF1E0(*a1, a2, v6, v12, a5, a6);
LABEL_18:
    v14 = (_QWORD *)(*(_QWORD *)(a1[1] + 272) + 16LL * v13);
    v9 |= *v14;
    v11 |= v14[1];
    goto LABEL_19;
  }
  v16 = *(_WORD *)(v17 + 68);
  if ( v16 > 0x14u )
    goto LABEL_19;
  v6 = 1LL << v16;
  if ( ((1LL << v16) & 0x180301) == 0
    || !sub_2DADC20(
          (_QWORD *)*a1,
          v17,
          *(_QWORD *)(*(_QWORD *)(*a1 + 56) + 16LL * (v15 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
          (_DWORD *)v7) )
  {
    goto LABEL_19;
  }
  if ( v13 )
    goto LABEL_18;
  return sub_2EBF1E0(*a1, a2, v6, v12, a5, a6);
}
