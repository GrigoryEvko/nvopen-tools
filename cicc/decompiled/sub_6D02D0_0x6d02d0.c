// Function: sub_6D02D0
// Address: 0x6d02d0
//
__int64 __fastcall sub_6D02D0(
        __m128i *a1,
        __m128i *a2,
        _QWORD *a3,
        __int64 *a4,
        _QWORD *a5,
        _QWORD *a6,
        unsigned int a7,
        int a8,
        unsigned int a9,
        unsigned int a10)
{
  __int16 v12; // r9
  unsigned int v13; // ecx
  __int64 v14; // r12
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v21; // rdx
  int v22; // eax
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rax
  unsigned int v29; // [rsp+8h] [rbp-58h]
  __int16 v30; // [rsp+Ch] [rbp-54h]
  __int64 *v31; // [rsp+10h] [rbp-50h]
  _QWORD *v32; // [rsp+10h] [rbp-50h]
  _QWORD *v33; // [rsp+18h] [rbp-48h]
  _QWORD *v34; // [rsp+18h] [rbp-48h]
  __int64 v35[7]; // [rsp+28h] [rbp-38h] BYREF

  v12 = a7;
  v13 = a9;
  if ( !a10 )
  {
    if ( a6 )
    {
      if ( a9 )
        return sub_6D0040(a7, a6, a4, a2, a1);
      else
        return sub_6CFEF0(a7, a6, a4, a2, a1->m128i_i64);
    }
    if ( !a8 )
      return sub_6E6260(a1);
    v35[0] = sub_724DC0(a10, a2, a3, a9, a5, a7);
    v23 = sub_72C390();
    sub_72BB40(v23, v35[0]);
    if ( (_WORD)a7 == 67 )
    {
      sub_6E6A50(v35[0], a1);
      v28 = sub_72CBE0(v35[0], a1, v24, v25, v26, v27);
      sub_6F7220(a1, v28);
      return sub_724E30(v35);
    }
    if ( (_WORD)a7 != 53 )
    {
      if ( (_WORD)a7 != 52 )
      {
        sub_6851C0(0xB2Bu, a4);
        sub_6E6260(a1);
        return sub_724E30(v35);
      }
      sub_620D80((_WORD *)(v35[0] + 176), 1);
    }
    sub_6E6A50(v35[0], a1);
    return sub_724E30(v35);
  }
  if ( dword_4F077C4 != 2 || unk_4F07778 <= 201702 )
  {
    if ( dword_4F077BC )
    {
      if ( !dword_4D03A18 )
      {
        v32 = a5;
        v34 = a3;
        v22 = sub_729F80(dword_4F063F8);
        a3 = v34;
        a5 = v32;
        v12 = a7;
        v13 = a9;
        if ( !v22 )
        {
          sub_684B30(0xB66u, a4);
          v13 = a9;
          v12 = a7;
          dword_4D03A18 = 1;
          a5 = v32;
          a3 = v34;
        }
      }
    }
  }
  v29 = v13;
  v30 = v12;
  v31 = a5;
  v33 = a3;
  v14 = sub_726700(30);
  *(_QWORD *)v14 = *(_QWORD *)&dword_4D03B80;
  *(_QWORD *)(v14 + 28) = *a4;
  v15 = v29 & 1;
  *(_QWORD *)(v14 + 36) = *v33;
  v16 = *v31;
  *(_WORD *)(v14 + 64) = v30;
  *(_QWORD *)(v14 + 44) = v16;
  *(_BYTE *)(v14 + 66) = v15 | *(_BYTE *)(v14 + 66) & 0xFE;
  v17 = sub_6F6D20(a6, 1, v15);
  *(_QWORD *)(v14 + 56) = v17;
  if ( a8 | v29 ^ 1 )
  {
    while ( v17 )
    {
      v18 = v17;
      v17 = *(_QWORD *)(v17 + 16);
      if ( !a8 && !v17 )
        break;
      v19 = *(_QWORD *)(v18 + 80);
      if ( v19 && !*(_QWORD *)(v19 + 128) )
        continue;
      *(_BYTE *)(v18 + 26) |= 4u;
    }
  }
  else
  {
    while ( 1 )
    {
      v17 = *(_QWORD *)(v17 + 16);
      if ( !v17 )
        break;
      while ( 1 )
      {
        v21 = *(_QWORD *)(v17 + 80);
        if ( v21 )
        {
          if ( !*(_QWORD *)(v21 + 128) )
            break;
        }
        *(_BYTE *)(v17 + 26) |= 4u;
        v17 = *(_QWORD *)(v17 + 16);
        if ( !v17 )
          goto LABEL_12;
      }
    }
  }
LABEL_12:
  sub_6E70E0(v14, a1);
  return sub_6E1990(a6);
}
