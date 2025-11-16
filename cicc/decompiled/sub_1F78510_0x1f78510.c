// Function: sub_1F78510
// Address: 0x1f78510
//
__int64 __fastcall sub_1F78510(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rbx
  __int16 v7; // dx
  __int64 v8; // r14
  __int16 v9; // dx
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v13; // rax
  bool v14; // zf
  __int64 v15; // rax
  int v16; // r15d
  __int64 v17; // rax
  char v18; // di
  __int64 v19; // rax
  int v20; // eax
  __int64 v21; // r9
  __int64 v22; // rdx
  unsigned int v23; // r14d
  __int64 v24; // r15
  __int64 v25; // rax
  __int64 v26; // rax
  _QWORD *v27; // r10
  __int64 v28; // r15
  __int64 *v29; // rbx
  __int64 v30; // [rsp+0h] [rbp-90h]
  _QWORD *v31; // [rsp+8h] [rbp-88h]
  __int64 v32; // [rsp+10h] [rbp-80h] BYREF
  __int64 v33; // [rsp+18h] [rbp-78h]
  char v34[8]; // [rsp+20h] [rbp-70h] BYREF
  __int64 v35; // [rsp+28h] [rbp-68h]
  __int64 v36; // [rsp+30h] [rbp-60h] BYREF
  int v37; // [rsp+38h] [rbp-58h]
  _QWORD v38[10]; // [rsp+40h] [rbp-50h] BYREF

  v5 = *(_QWORD *)(a2 + 32);
  v32 = a3;
  v33 = a4;
  v6 = *(_QWORD *)v5;
  v7 = *(_WORD *)(*(_QWORD *)v5 + 24LL);
  if ( v7 == 51 )
  {
    v6 = *(_QWORD *)(*(_QWORD *)(v6 + 32) + 40LL * *(unsigned int *)(v5 + 8));
    v7 = *(_WORD *)(v6 + 24);
  }
  v8 = *(_QWORD *)(v5 + 40);
  if ( v7 != 185 )
    v6 = 0;
  v9 = *(_WORD *)(v8 + 24);
  if ( v9 == 51 )
  {
    v10 = *(_QWORD *)(v8 + 32) + 40LL * *(unsigned int *)(v5 + 48);
    v8 = *(_QWORD *)v10;
    v9 = *(_WORD *)(*(_QWORD *)v10 + 24LL);
  }
  v11 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  if ( v9 != 185 )
  {
    sub_1E0A0C0(v11);
    return 0;
  }
  if ( *(_BYTE *)sub_1E0A0C0(v11) )
  {
    if ( !v6 )
      return 0;
    v13 = v6;
    v6 = v8;
    v14 = *(_WORD *)(v8 + 24) == 185;
    v8 = v13;
    if ( !v14 )
      return 0;
  }
  else if ( !v6 || *(_WORD *)(v6 + 24) != 185 )
  {
    return 0;
  }
  if ( (*(_BYTE *)(v6 + 27) & 0xC) != 0 )
    return 0;
  v15 = *(_QWORD *)(v6 + 48);
  if ( !v15 )
    return 0;
  if ( *(_QWORD *)(v15 + 32) )
    return 0;
  v16 = sub_1E340A0(*(_QWORD *)(v6 + 104));
  if ( (unsigned int)sub_1E340A0(*(_QWORD *)(v8 + 104)) != v16 )
    return 0;
  v17 = *(_QWORD *)(v6 + 40);
  v18 = *(_BYTE *)v17;
  v19 = *(_QWORD *)(v17 + 8);
  v34[0] = v18;
  v35 = v19;
  v20 = v18 ? sub_1F6C8D0(v18) : sub_1F58D40((__int64)v34);
  if ( *(_WORD *)(v8 + 24) != 185 )
    return 0;
  if ( (*(_BYTE *)(v8 + 27) & 0xC) != 0 )
    return 0;
  v22 = *(_QWORD *)(v8 + 48);
  if ( !v22 )
    return 0;
  if ( *(_QWORD *)(v22 + 32) )
    return 0;
  if ( !sub_1D19900(*(_QWORD *)a1, v8, v6, (unsigned int)(v20 + 7) >> 3, 1, v21) )
    return 0;
  v23 = sub_1E34390(*(_QWORD *)(v6 + 104));
  v24 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)a1 + 32LL));
  v25 = sub_1F58E60((__int64)&v32, *(_QWORD **)(*(_QWORD *)a1 + 48LL));
  if ( (unsigned int)sub_15A9FE0(v24, v25) > v23 || *(_BYTE *)(a1 + 24) && !sub_1F6C830(*(_QWORD *)(a1 + 8), 0xB9u, v32) )
    return 0;
  v26 = *(_QWORD *)(a2 + 72);
  v27 = *(_QWORD **)a1;
  memset(v38, 0, 24);
  v28 = *(_QWORD *)(v6 + 104);
  v29 = *(__int64 **)(v6 + 32);
  v36 = v26;
  if ( v26 )
  {
    v31 = v27;
    sub_1F6CA20(&v36);
    v27 = v31;
  }
  v37 = *(_DWORD *)(a2 + 64);
  v30 = sub_1D2B730(
          v27,
          (unsigned int)v32,
          v33,
          (__int64)&v36,
          *v29,
          v29[1],
          v29[5],
          v29[6],
          *(_OWORD *)v28,
          *(_QWORD *)(v28 + 16),
          v23,
          0,
          (__int64)v38,
          0);
  sub_17CD270(&v36);
  return v30;
}
