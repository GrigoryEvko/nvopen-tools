// Function: sub_600CA0
// Address: 0x600ca0
//
__int64 __fastcall sub_600CA0(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // r13
  __int64 v5; // r12
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // r12
  __int64 v10; // r14
  char v11; // al
  __int64 v12; // r8
  char v13; // al
  __int64 v14; // rdx
  __int64 *v15; // r13
  char v16; // al
  __int64 v17; // rax
  __int64 v18; // rbx
  __int64 v19; // rax
  char v21; // al
  __int64 v22; // r15
  char v23; // cl
  __int64 v24; // rax
  __int64 v25; // r12
  _BYTE *v26; // rax
  __int64 v27; // [rsp+10h] [rbp-370h]
  __int64 v28; // [rsp+18h] [rbp-368h]
  __int64 v29; // [rsp+20h] [rbp-360h]
  int v30; // [rsp+38h] [rbp-348h]
  char i; // [rsp+3Fh] [rbp-341h]
  __m128i v33[4]; // [rsp+50h] [rbp-330h] BYREF
  char v34[8]; // [rsp+90h] [rbp-2F0h] BYREF
  char v35[56]; // [rsp+98h] [rbp-2E8h] BYREF
  char v36; // [rsp+D0h] [rbp-2B0h]
  char v37; // [rsp+D1h] [rbp-2AFh]
  __int64 v38; // [rsp+100h] [rbp-280h] BYREF
  __int64 v39; // [rsp+108h] [rbp-278h]
  char v40; // [rsp+17Fh] [rbp-201h]
  char v41; // [rsp+182h] [rbp-1FEh]
  char v42; // [rsp+183h] [rbp-1FDh]
  __int64 v43; // [rsp+220h] [rbp-160h]
  __int64 v44; // [rsp+240h] [rbp-140h]
  char v45; // [rsp+330h] [rbp-50h]

  v3 = *(_QWORD *)(a1 + 88);
  v4 = a2[3];
  v5 = *(_QWORD *)(v3 + 152);
  for ( i = *(_BYTE *)(a3 + 12); *(_BYTE *)(v5 + 140) == 12; v5 = *(_QWORD *)(v5 + 160) )
    ;
  v30 = sub_72F3C0(v5, a2[6], 0, 1, 1);
  if ( !v30 || (LOBYTE(v30) = 1, ***(_QWORD ***)(v5 + 168)) )
  {
    v6 = *(_QWORD *)a3;
    v29 = *(_QWORD *)a3;
    v7 = sub_7259C0(7);
    v8 = *(_QWORD *)(v7 + 168);
    v9 = v7;
    sub_73BCD0(*(_QWORD *)(v3 + 152), v7, 0);
    *(_BYTE *)(v8 + 21) |= 1u;
    *(_QWORD *)(v8 + 8) = 0;
    *(_QWORD *)(v8 + 56) = 0;
    *(_QWORD *)(v8 + 40) = v6;
    v28 = *(_QWORD *)v6;
    v10 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v6 + 96LL) + 8LL);
    if ( !v10 )
      goto LABEL_8;
    v11 = *(_BYTE *)(v10 + 80);
    if ( v11 != 17 )
      goto LABEL_39;
    v10 = *(_QWORD *)(v10 + 88);
    if ( !v10 )
    {
LABEL_8:
      sub_5E4C60((__int64)&v38, a2 + 1);
      v45 |= 2u;
      v41 |= 0x80u;
      v44 = v3;
      if ( (*(_BYTE *)(v4 + 96) & 2) != 0 || *(char *)(v3 + 194) < 0 )
        v42 |= 1u;
      v40 |= 0x10u;
      v43 = v9;
      v13 = *(_BYTE *)(v3 + 193);
      if ( (v13 & 2) != 0 )
        v39 |= 0x80000uLL;
      if ( (v13 & 4) != 0 )
        v39 |= 0x100000uLL;
      v27 = v12;
      sub_87E3B0(v34);
      v36 |= 2u;
      sub_878710(v28, v33);
      sub_87A680(v33, v27, 0);
      *(_BYTE *)(a3 + 12) = *(_BYTE *)(v3 + 88) & 3;
      sub_5FBCD0(v33, (__int64)v34, a3, &v38, 1u);
      v14 = v38;
      v15 = *(__int64 **)(v38 + 88);
      v15[45] = (__int64)a2;
      v16 = *((_BYTE *)v15 + 194) | 0x40;
      *((_BYTE *)v15 + 194) = v16;
      if ( *(char *)(v3 + 193) < 0 )
        *((_BYTE *)v15 + 193) |= 0x80u;
      if ( (*(_BYTE *)(v3 + 194) & 0x10) != 0 )
      {
        *((_BYTE *)v15 + 194) = v16 | 0x10;
        *(_BYTE *)(*(_QWORD *)(v29 + 168) + 110LL) |= 2u;
      }
      v15[31] = *(_QWORD *)(v3 + 248);
      *(_QWORD *)(v14 + 96) = *(_QWORD *)(a1 + 96);
      if ( dword_4D048B8 )
      {
        v25 = *(_QWORD *)(v9 + 168);
        v26 = (_BYTE *)sub_725E60(v33, v34, dword_4D048B8);
        *v26 |= 0xAu;
        *(_QWORD *)(v25 + 56) = v26;
        *(_QWORD *)(v25 + 8) = v15;
      }
      if ( dword_4F04C64 == -1
        || (v17 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v17 + 7) & 1) == 0)
        || dword_4F04C44 == -1 && (*(_BYTE *)(v17 + 6) & 2) == 0 )
      {
        if ( (v37 & 8) == 0 )
          sub_87E280(v35);
      }
      if ( dword_4F068EC )
      {
        if ( (*((_BYTE *)v15 + 206) & 0x10) != 0 )
          goto LABEL_31;
        if ( (*((_BYTE *)v15 + 193) & 4) != 0 )
        {
LABEL_29:
          if ( (*(_BYTE *)(v3 + 206) & 0x10) != 0 || (unsigned int)sub_600C10(v29) )
          {
            *((_BYTE *)v15 + 206) |= 0x10u;
            *((_BYTE *)v15 + 193) |= 0x20u;
          }
          goto LABEL_31;
        }
        sub_89A080(v15);
      }
      if ( (*((_BYTE *)v15 + 206) & 0x10) == 0 )
        goto LABEL_29;
LABEL_31:
      v18 = *v15;
      v19 = sub_5E4B20(v29);
      *(_QWORD *)(v19 + 16) = v18;
      *(_BYTE *)(v19 + 184) = *(_BYTE *)(v19 + 184) & 0xCF | (32 * v30) & 0x20 | 0x10;
      sub_5E9580(v19);
      goto LABEL_32;
    }
    while ( 1 )
    {
      v11 = *(_BYTE *)(v10 + 80);
LABEL_39:
      if ( v11 != 10 )
        goto LABEL_37;
      v22 = *(_QWORD *)(v10 + 88);
      v21 = *(_BYTE *)(v22 + 194);
      if ( (v21 & 0x40) != 0 )
      {
        if ( (*(_BYTE *)(v4 + 96) & 2) == 0 )
          goto LABEL_36;
      }
      else
      {
        if ( (unsigned int)sub_8DED30(*(_QWORD *)(v22 + 152), v9, 1048708)
          && (unsigned int)sub_739400(*(_QWORD *)(v22 + 216), *(_QWORD *)(v3 + 216)) )
        {
          break;
        }
        v21 = *(_BYTE *)(v22 + 194);
        if ( (v21 & 0x40) == 0 )
          goto LABEL_37;
        if ( (*(_BYTE *)(v4 + 96) & 2) == 0 )
        {
LABEL_36:
          if ( *(char *)(v3 + 194) >= 0 )
            goto LABEL_37;
        }
      }
      if ( v21 < 0 )
      {
        v23 = *(_BYTE *)(v3 + 194);
        v24 = v3;
        if ( (v23 & 0x40) != 0 )
        {
          do
            v24 = *(_QWORD *)(v24 + 232);
          while ( (v23 >= 0 || *(char *)(v24 + 194) < 0) && (*(_BYTE *)(v24 + 194) & 0x40) != 0 );
        }
        do
          v22 = *(_QWORD *)(v22 + 232);
        while ( *(char *)(v22 + 194) < 0 && (*(_BYTE *)(v22 + 194) & 0x40) != 0 );
        if ( v22 == v24 )
          break;
      }
LABEL_37:
      v10 = *(_QWORD *)(v10 + 8);
      if ( !v10 )
        goto LABEL_8;
    }
  }
LABEL_32:
  *(_BYTE *)(a3 + 12) = i;
  return a3;
}
