// Function: sub_732910
// Address: 0x732910
//
__int16 __fastcall sub_732910(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r12d
  __int64 v7; // r14
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // rdi
  _BYTE *v14; // rax
  __int64 v15; // r12

  v6 = a3;
  v7 = *(_QWORD *)(a1 + 248);
  if ( v7 )
  {
    if ( (unsigned int)sub_825090(*(_QWORD *)(a1 + 248)) )
    {
      sub_8250A0(v7);
    }
    else if ( (unsigned int)sub_8250B0(v7) )
    {
      sub_8250C0(v7);
    }
  }
  if ( !v6 )
  {
    v9 = *(_QWORD *)(a1 + 152);
    for ( *(_BYTE *)(a1 + 88) |= 4u; *(_BYTE *)(v9 + 140) == 12; v9 = *(_QWORD *)(v9 + 160) )
      ;
    if ( *(_QWORD *)(*(_QWORD *)(v9 + 168) + 40LL) )
      *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL) + 88LL) |= 4u;
  }
  sub_71D150(a1, a2, a3, a4, a5, a6);
  v13 = *(_QWORD *)(a1 + 344);
  if ( v13 )
    sub_5EB240(v13);
  v14 = &dword_4F04C44;
  if ( dword_4F04C44 != -1
    || (v11 = qword_4F04C68[0],
        v10 = dword_4F04C64,
        v14 = (_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64),
        (v14[6] & 2) != 0)
    || dword_4F04C64 != -1 && (v14[14] & 2) != 0
    || !(_DWORD)a2
    || unk_4D03FD8
    && (unk_4F04C48 == -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) == 0)
    && dword_4F04C58 != -1
    && (v14 = *(_BYTE **)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 216), (char)v14[192] >= 0) )
  {
    v15 = *(_QWORD *)a1;
    if ( !*(_QWORD *)a1 )
      return (__int16)v14;
    goto LABEL_22;
  }
  LOWORD(v14) = *(_WORD *)(a1 + 192) & 0x2480;
  if ( (_WORD)v14 == 128 )
  {
    LODWORD(v14) = sub_825070(a1);
    if ( (_DWORD)v14 )
      LOWORD(v14) = sub_825080(a1);
  }
  v15 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    a2 = 1;
    sub_8AD0D0(*(_QWORD *)a1, 1, 0);
LABEL_22:
    LOWORD(v14) = sub_894C00(v15, a2, v10, v11, v12);
  }
  return (__int16)v14;
}
