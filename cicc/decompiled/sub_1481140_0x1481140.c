// Function: sub_1481140
// Address: 0x1481140
//
__int64 __fastcall sub_1481140(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 v9; // rbx
  __int64 v10; // r8
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // rax
  _QWORD *v14; // rdi
  _QWORD *v15; // rsi
  bool v16; // zf
  _QWORD *v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rbx
  _QWORD *v20; // rdi
  _QWORD *v21; // rsi
  __int64 v22; // rbx
  int v23; // eax
  __int64 v24; // [rsp+0h] [rbp-50h]
  __int64 v25; // [rsp+0h] [rbp-50h]
  __int64 v26; // [rsp+8h] [rbp-48h]
  __int64 v27; // [rsp+8h] [rbp-48h]
  __int64 v28[7]; // [rsp+18h] [rbp-38h] BYREF

  if ( (unsigned __int8)sub_1480BD0(a1, a2, a3, a4) )
    return 1;
  if ( a2 == 39 )
  {
    v27 = a3;
    v19 = a4;
LABEL_18:
    v25 = sub_14561E0(v19);
    if ( v25 )
    {
      v28[0] = sub_1480810(a1, v27);
      if ( *(_WORD *)(v25 + 24) == 9 )
      {
        v20 = *(_QWORD **)(v25 + 32);
        v21 = &v20[*(_QWORD *)(v25 + 40)];
        if ( v21 != sub_1453020(v20, (__int64)v21, v28) )
          return 1;
      }
    }
    v28[0] = v19;
    if ( *(_WORD *)(v27 + 24) != 9 )
      goto LABEL_14;
    v17 = *(_QWORD **)(v27 + 32);
    v18 = *(_QWORD *)(v27 + 40);
    goto LABEL_13;
  }
  if ( a2 > 0x27 )
  {
    if ( a2 != 41 )
      goto LABEL_24;
    v27 = a4;
    v19 = a3;
    goto LABEL_18;
  }
  if ( a2 == 35 )
  {
    v9 = a3;
    v10 = a4;
  }
  else
  {
    if ( a2 != 37 )
      goto LABEL_14;
    v9 = a4;
    v10 = a3;
  }
  v26 = v10;
  v11 = sub_14561E0(v10);
  v12 = v26;
  v24 = v11;
  if ( v11 )
  {
    v13 = sub_1480810(a1, v9);
    v12 = v26;
    v28[0] = v13;
    if ( *(_WORD *)(v24 + 24) == 8 )
    {
      v14 = *(_QWORD **)(v24 + 32);
      v15 = &v14[*(_QWORD *)(v24 + 40)];
      if ( v15 != sub_1453020(v14, (__int64)v15, v28) )
        return 1;
    }
  }
  v16 = *(_WORD *)(v9 + 24) == 8;
  v28[0] = v12;
  if ( v16 )
  {
    v17 = *(_QWORD **)(v9 + 32);
    v18 = *(_QWORD *)(v9 + 40);
LABEL_13:
    if ( &v17[v18] != sub_1453020(v17, (__int64)&v17[v18], v28) )
      return 1;
  }
LABEL_14:
  if ( a2 - 32 > 1 )
  {
LABEL_24:
    if ( *(_WORD *)(a3 + 24) != 7 )
      return sub_1457940(a1, a2, a3, a4);
    if ( *(_WORD *)(a4 + 24) != 7 )
      return sub_1457940(a1, a2, a3, a4);
    if ( *(_QWORD *)(a3 + 48) != *(_QWORD *)(a4 + 48) )
      return sub_1457940(a1, a2, a3, a4);
    if ( *(_QWORD *)(a3 + 40) != 2 )
      return sub_1457940(a1, a2, a3, a4);
    if ( *(_QWORD *)(a4 + 40) != 2 )
      return sub_1457940(a1, a2, a3, a4);
    v22 = sub_13A5BC0((_QWORD *)a3, a1);
    if ( v22 != sub_13A5BC0((_QWORD *)a4, a1) )
      return sub_1457940(a1, a2, a3, a4);
    v23 = (unsigned __int8)sub_15FF7F0(a2) == 0 ? 2 : 4;
    if ( ((unsigned __int16)v23 & *(_WORD *)(a3 + 26)) == 0
      || ((unsigned __int16)v23 & *(_WORD *)(a4 + 26)) == 0
      || !(unsigned __int8)sub_147A340(a1, a2, **(_QWORD **)(a3 + 32), **(_QWORD **)(a4 + 32)) )
    {
      return sub_1457940(a1, a2, a3, a4);
    }
    return 1;
  }
  return sub_1457940(a1, a2, a3, a4);
}
