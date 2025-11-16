// Function: sub_7CFE40
// Address: 0x7cfe40
//
__int64 __fastcall sub_7CFE40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rax
  __int64 *v7; // r13
  _BOOL4 v8; // r14d
  __int64 v10; // r15
  _QWORD *v11; // rax
  __int64 v12; // r12
  char v13; // al
  char v14; // dl
  __int64 i; // rax
  __int64 v16; // r15

  v6 = *(_QWORD **)(a1 + 168);
  if ( *(_BYTE *)(a1 + 140) == 14 )
  {
    v7 = *(__int64 **)(a1 + 168);
    v8 = (*(_BYTE *)(a1 + 161) & 2) != 0;
    if ( *v6 )
      return *v7;
  }
  else
  {
    v7 = v6 + 4;
    v8 = 0;
    if ( v6[4] )
      return *v7;
  }
  if ( *(_QWORD *)a1 )
    v10 = sub_87EBB0(4, **(_QWORD **)a1);
  else
    v10 = sub_87F680(4, &dword_4F077C8, a3, a4, a5, a6);
  *(_DWORD *)(v10 + 40) = unk_4F066A8;
  v11 = sub_7259C0(9);
  v12 = (__int64)v11;
  if ( !v8 )
  {
    *((_BYTE *)v11 + 141) &= ~0x20u;
    v11[16] = 1;
    *((_DWORD *)v11 + 34) = 1;
  }
  sub_877D80(v11, v10);
  v13 = *(_BYTE *)(a1 + 90) & 0x10 | *(_BYTE *)(v12 + 90) & 0xEF;
  *(_BYTE *)(v12 + 90) = v13;
  v14 = *(_BYTE *)(a1 + 90);
  *(_BYTE *)(v12 + 178) |= 0x20u;
  *(_BYTE *)(v12 + 90) = v14 & 0x20 | v13 & 0xDF;
  *(_QWORD *)(v10 + 88) = v12;
  if ( (*(_BYTE *)(a1 + 89) & 4) != 0 )
    sub_877E20(v10, v12, *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL));
  for ( i = v12; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v16 = *(_QWORD *)(*(_QWORD *)i + 96LL);
  *(_DWORD *)(v16 + 96) = sub_880E90();
  *(_BYTE *)(v12 + 177) = (32 * !v8) | *(_BYTE *)(v12 + 177) & 0xDF;
  if ( dword_4F07590 | v8 )
    sub_7365B0(v12, 0);
  *v7 = v12;
  *(_QWORD *)(*(_QWORD *)(v12 + 168) + 256LL) = a1;
  return *v7;
}
