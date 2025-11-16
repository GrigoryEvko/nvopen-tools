// Function: sub_F29A20
// Address: 0xf29a20
//
_QWORD *__fastcall sub_F29A20(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  _QWORD *v3; // r14
  __int64 v4; // rax
  __int64 v5; // r8
  __int64 v6; // rbx
  int v7; // eax
  __int64 v8; // r12
  __int64 v9; // r13
  _QWORD *v10; // rax
  char v12; // al
  __int64 v13; // rdx
  char v14; // al
  __int64 v15; // rax
  __int64 v16; // r9
  __int64 v17; // rbx
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // r12
  __int64 v21; // r12
  __int64 v22; // rax
  __int64 v23; // [rsp+10h] [rbp-50h]
  __int64 v24; // [rsp+10h] [rbp-50h]
  __int64 v25; // [rsp+18h] [rbp-48h]
  __int64 v26[7]; // [rsp+28h] [rbp-38h] BYREF

  if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) == 0 )
    return 0;
  v2 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  if ( !v2 )
    return 0;
  v3 = (_QWORD *)a2;
  v4 = sub_B43CB0(a2);
  v5 = *(_QWORD *)(v2 + 8);
  v6 = v4;
  if ( *(_BYTE *)(v5 + 8) != 14 )
    goto LABEL_4;
  v23 = *(_QWORD *)(v2 + 8);
  v26[0] = *(_QWORD *)(v4 + 120);
  v25 = sub_A74620(v26);
  v12 = sub_B2D630(v6, 43);
  v5 = v23;
  if ( !v12 )
  {
    if ( !v25 )
      goto LABEL_4;
    v13 = v23;
    if ( (unsigned int)*(unsigned __int8 *)(v23 + 8) - 17 <= 1 )
      v13 = **(_QWORD **)(v23 + 16);
    v14 = sub_B2F070(v6, *(_DWORD *)(v13 + 8) >> 8);
    v5 = v23;
    if ( v14 )
    {
LABEL_4:
      if ( (unsigned __int8)sub_A750C0(v5) )
      {
        v26[0] = *(_QWORD *)(v6 + 120);
        v7 = sub_A74660(v26);
        if ( v7 )
        {
          v26[0] = 1023;
          v8 = sub_11A3690(a1, v2, ~(_WORD)v7 & 0x3FF, v26, 0, a2);
          if ( v8 )
          {
            v9 = sub_BD5C60(a2);
            v10 = sub_BD2C40(72, 1u);
            v3 = v10;
            if ( v10 )
              sub_B4BB80((__int64)v10, v9, v8, 1u, 0, 0);
            return v3;
          }
        }
      }
      return 0;
    }
  }
  v24 = v5;
  v15 = sub_114CED0(a1, v2, v25 != 0, 0);
  if ( !v15 )
  {
    v5 = v24;
    goto LABEL_4;
  }
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v16 = *(_QWORD *)(a2 - 8);
  else
    v16 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v17 = *(_QWORD *)v16;
  if ( *(_QWORD *)v16 )
  {
    v18 = *(_QWORD *)(v16 + 8);
    **(_QWORD **)(v16 + 16) = v18;
    if ( v18 )
      *(_QWORD *)(v18 + 16) = *(_QWORD *)(v16 + 16);
  }
  *(_QWORD *)v16 = v15;
  v19 = *(_QWORD *)(v15 + 16);
  *(_QWORD *)(v16 + 8) = v19;
  if ( v19 )
    *(_QWORD *)(v19 + 16) = v16 + 8;
  *(_QWORD *)(v16 + 16) = v15 + 16;
  *(_QWORD *)(v15 + 16) = v16;
  if ( *(_BYTE *)v17 > 0x1Cu )
  {
    v20 = *(_QWORD *)(a1 + 40);
    v26[0] = v17;
    v21 = v20 + 2096;
    sub_F200C0(v21, v26);
    v22 = *(_QWORD *)(v17 + 16);
    if ( v22 )
    {
      if ( !*(_QWORD *)(v22 + 8) )
      {
        v26[0] = *(_QWORD *)(v22 + 24);
        sub_F200C0(v21, v26);
      }
    }
  }
  return v3;
}
