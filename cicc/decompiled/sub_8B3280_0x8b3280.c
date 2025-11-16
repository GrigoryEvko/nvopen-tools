// Function: sub_8B3280
// Address: 0x8b3280
//
__int64 __fastcall sub_8B3280(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, unsigned int a5)
{
  __int64 i; // rax
  __int64 v7; // r13
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r12
  __int64 v14; // r15
  __int64 v15; // r12
  __int64 v16; // r8
  __int64 v17; // rcx
  __int64 v18; // rsi
  __int64 *v19; // rdx
  __int64 j; // rax
  __int64 v21; // rax
  __int64 v22; // r14
  __int64 v23; // rdi
  __int64 v24; // rsi
  __int64 v25; // rdi

  for ( i = a2; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v7 = *(_QWORD *)(*(_QWORD *)i + 96LL);
  v8 = sub_892920(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 96LL) + 72LL));
  v9 = sub_892920(*(_QWORD *)(v7 + 72));
  v12 = v9;
  if ( *(_QWORD *)(v7 + 72) && sub_88FB10(v8, v9) && (*(_BYTE *)(a2 + 177) & 0x20) != 0 )
  {
    v14 = *(_QWORD *)(*(_QWORD *)(a1 + 168) + 168LL);
    v15 = *(_QWORD *)(*(_QWORD *)(a2 + 168) + 168LL);
    if ( (unsigned int)sub_88D900(v15) )
    {
      v16 = a5;
      v17 = a4;
      v18 = v15;
      v19 = a3;
      return (unsigned int)sub_8B4AF0(v14, v18, v19, v17, v16) != 0;
    }
    return 1;
  }
  if ( v12 && (*(_BYTE *)(*(_QWORD *)(v12 + 88) + 266LL) & 1) != 0 )
  {
    for ( j = a1; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    v21 = sub_892920(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)j + 96LL) + 72LL));
    if ( v21
      && (unsigned int)sub_8B30E0(
                         **(_QWORD **)(*(_QWORD *)(v21 + 88) + 104LL),
                         **(_QWORD **)(*(_QWORD *)(v12 + 88) + 104LL),
                         a3,
                         a4) )
    {
      v14 = *(_QWORD *)(*(_QWORD *)(a1 + 168) + 168LL);
      v22 = *(_QWORD *)(*(_QWORD *)(a2 + 168) + 168LL);
      if ( (unsigned int)sub_88D900(v22) )
      {
        v16 = a5;
        v17 = a4;
        v18 = v22;
        v19 = a3;
        LOBYTE(v16) = a5 | 0x80;
        return (unsigned int)sub_8B4AF0(v14, v18, v19, v17, v16) != 0;
      }
      return 1;
    }
    return 0;
  }
  if ( a2 == a1 || (unsigned int)sub_8D97D0(a1, a2, 0, v10, v11) )
    return 1;
  if ( (*(_BYTE *)(a2 + 89) & 4) == 0 )
    return 0;
  if ( unk_4D04318 )
  {
    if ( (*(_BYTE *)(a1 + 89) & 4) != 0 && **(_QWORD **)a1 == **(_QWORD **)a2 )
    {
      v23 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
      v24 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 32LL) + 168LL) + 256LL);
      if ( !v24 )
        v24 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 32LL);
      return (unsigned int)sub_8B3500(v23, v24, a3, a4, 0) != 0;
    }
    return 0;
  }
  v25 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 32LL) + 168LL) + 256LL);
  if ( !v25 )
    v25 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 32LL);
  return sub_8DC1A0(v25);
}
