// Function: sub_5EDE90
// Address: 0x5ede90
//
__int64 __fastcall sub_5EDE90(unsigned int a1, int a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // r15
  char v12; // al
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  char v19; // al
  __int64 v20; // r9
  int v21; // eax
  __int64 v22; // r9
  char v23; // al
  __int64 v24; // rdi
  __int64 v25; // r8
  __int64 v27; // rax
  __int64 v28; // rdx
  int v29; // ecx
  __int64 v30; // rax
  __int64 v31; // rbx
  char v32; // di
  __int64 v33; // rax
  __int64 v34; // [rsp+8h] [rbp-78h]
  _BOOL4 v36; // [rsp+20h] [rbp-60h]
  __int64 v37; // [rsp+20h] [rbp-60h]
  __int64 v38; // [rsp+20h] [rbp-60h]
  __int64 v39; // [rsp+28h] [rbp-58h]
  __int64 v40; // [rsp+30h] [rbp-50h]
  unsigned int v43[13]; // [rsp+4Ch] [rbp-34h] BYREF

  v34 = a3[3];
  v39 = *(_QWORD *)(a4 + 288);
  v40 = *(_QWORD *)(a5 + 72);
  sub_7296C0(v43);
  if ( a2 )
  {
    v9 = sub_87EBB0(10, *a3);
    *(_QWORD *)a4 = v9;
    v10 = v9;
    v11 = sub_646F50(v39, 1, 0xFFFFFFFFLL);
    sub_7362F0(v11, (unsigned int)dword_4F04C64);
    v12 = *(_BYTE *)(a4 + 125);
    *(_BYTE *)(v11 + 195) |= 0x2Au;
    *(_BYTE *)(v11 + 207) = (2 * v12) & 0x10 | *(_BYTE *)(v11 + 207) & 0xEF;
    v36 = 1;
    goto LABEL_3;
  }
  v36 = (*((_BYTE *)a3 + 18) & 2) != 0;
  v27 = sub_87EBB0((unsigned int)((*((_BYTE *)a3 + 18) >> 1 << 31 >> 31) + 11), *a3);
  *(_QWORD *)a4 = v27;
  v10 = v27;
  v11 = sub_646F50(v39, 1, 0xFFFFFFFFLL);
  if ( dword_4F07590 )
    goto LABEL_51;
  if ( unk_4F04C44 != -1 )
  {
    *(_BYTE *)(v11 + 207) = (2 * *(_BYTE *)(a4 + 125)) & 0x10 | *(_BYTE *)(v11 + 207) & 0xEF;
    goto LABEL_39;
  }
  v28 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( (*(_BYTE *)(v28 + 6) & 6) != 0 || *(_BYTE *)(v28 + 4) == 12 )
  {
    *(_BYTE *)(v11 + 207) = (2 * *(_BYTE *)(a4 + 125)) & 0x10 | *(_BYTE *)(v11 + 207) & 0xEF;
  }
  else
  {
LABEL_51:
    sub_7362F0(v11, unk_4F04C34);
    v29 = unk_4F04C44;
    *(_BYTE *)(v11 + 207) = (2 * *(_BYTE *)(a4 + 125)) & 0x10 | *(_BYTE *)(v11 + 207) & 0xEF;
    if ( v29 != -1 )
      goto LABEL_39;
  }
  v30 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( (*(_BYTE *)(v30 + 6) & 6) != 0 || *(_BYTE *)(v30 + 4) == 12 )
  {
LABEL_39:
    *(_BYTE *)(v11 + 195) |= 8u;
    v13 = *(_QWORD *)(a4 + 8);
    if ( (v13 & 0x100000) != 0 )
      goto LABEL_4;
    goto LABEL_40;
  }
LABEL_3:
  v13 = *(_QWORD *)(a4 + 8);
  if ( (v13 & 0x100000) != 0 )
  {
LABEL_4:
    *(_BYTE *)(v11 + 193) |= 6u;
    goto LABEL_5;
  }
LABEL_40:
  if ( (v13 & 0x80000) != 0 )
    *(_BYTE *)(v11 + 193) |= 3u;
LABEL_5:
  *(_QWORD *)(v10 + 88) = v11;
  v14 = v10;
  sub_877D80(v11, v10);
  if ( (*(_BYTE *)(a4 + 16) & 0x10) != 0 )
  {
    v14 = 1;
    sub_725ED0(v11, 1);
  }
  else
  {
    v19 = *((_BYTE *)a3 + 16);
    if ( (v19 & 0x20) != 0 )
    {
      v14 = 2;
      sub_725ED0(v11, 2);
      if ( v36 )
        goto LABEL_11;
      goto LABEL_33;
    }
    if ( (v19 & 8) != 0 )
    {
      v14 = 5;
      sub_725ED0(v11, 5);
      *(_BYTE *)(v11 + 176) = *((_BYTE *)a3 + 56);
    }
    else if ( (v19 & 0x10) != 0 )
    {
      v14 = 3;
      sub_725ED0(v11, 3);
    }
  }
  if ( v36 )
  {
LABEL_11:
    if ( a2 )
    {
      v20 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 208);
    }
    else
    {
      v20 = 0;
      if ( (*((_BYTE *)a3 + 18) & 2) != 0 )
        v20 = a3[4];
    }
    v37 = v20;
    v21 = sub_8D3D40(v20);
    v22 = v37;
    if ( v21 )
    {
      while ( *(_BYTE *)(v22 + 140) == 12 )
        v22 = *(_QWORD *)(v22 + 160);
      v22 = sub_7CFE40(v22);
    }
    v14 = v11;
    v38 = v22;
    sub_877E20(v10, v11, v22);
    v18 = v38;
    if ( !a1 || !dword_4F077BC || !v34 || (*(_BYTE *)(v34 + 84) & 2) == 0 )
      goto LABEL_18;
    v14 = v34;
    nullsub_2();
    v23 = *((_BYTE *)a3 + 18);
    goto LABEL_19;
  }
LABEL_33:
  v23 = *((_BYTE *)a3 + 18);
  if ( (v23 & 2) == 0 )
  {
    v15 = a3[4];
    if ( v15 )
    {
      v14 = v11;
      sub_877E90(v10, v11);
LABEL_18:
      v23 = *((_BYTE *)a3 + 18);
    }
  }
LABEL_19:
  if ( (v23 & 1) != 0 )
  {
    v24 = a3[5];
    if ( v24 )
    {
      sub_697FC0(v24, v14, v15, v16, v17, v18);
      *(_QWORD *)(v11 + 240) = a3[5];
    }
    *(_BYTE *)(v11 + 203) |= 0x10u;
    *(_BYTE *)(v11 + 195) |= 1u;
  }
  if ( (*(_BYTE *)(a5 + 64) & 4) != 0 )
  {
    sub_736C90(v11, 1);
    *(_BYTE *)(v10 + 81) |= 2u;
    *(_BYTE *)(v11 + 193) |= 0x20u;
    if ( a1 )
      *(_BYTE *)(v11 + 202) |= 0x80u;
    *(_QWORD *)(v11 + 264) = *(_QWORD *)(a5 + 80);
    *(_BYTE *)(v11 + 173) = *(_BYTE *)(a4 + 268);
    *(_QWORD *)(v11 + 216) = *(_QWORD *)(a4 + 400);
    if ( v40 && dword_4F07590 )
    {
      *(_BYTE *)(v40 + 16) = 11;
      *(_QWORD *)(v40 + 24) = v11;
    }
    *(_QWORD *)(v11 + 72) = sub_729420(*(_BYTE *)(v11 - 8) & 1, a6);
  }
  else
  {
    v25 = a1;
    if ( a1 && unk_4D0433C && !unk_4D04340 && (a3[2] & 0x10001) == 0 && (unsigned int)sub_8DBE70(v39) )
      sub_685490(1301, a3 + 1, v10);
    if ( v40 )
    {
      v16 = dword_4F07590;
      if ( dword_4F07590 )
      {
        v31 = sub_86A1D0(v11, 11, *(_QWORD *)(a5 + 80));
        v32 = *(_BYTE *)(v31 - 8);
        *(_QWORD *)v31 = *(_QWORD *)(v10 + 48);
        v33 = sub_729420(v32 & 1, a6);
        v15 = a1;
        *(_QWORD *)(v31 + 8) = v33;
        if ( a1 )
          *(_BYTE *)(v31 + 57) |= 4u;
        if ( a2 )
          *(_BYTE *)(v31 + 57) |= 0x10u;
        *(_BYTE *)(v40 + 16) = 53;
        *(_QWORD *)(v40 + 24) = v31;
      }
    }
  }
  sub_644920(a4, (*(_BYTE *)(a5 + 64) & 4) != 0, v15, v16, v25, v18);
  sub_729730(v43[0]);
  return v10;
}
