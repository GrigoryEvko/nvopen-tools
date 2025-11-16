// Function: sub_12B27B0
// Address: 0x12b27b0
//
__int64 __fastcall sub_12B27B0(__int64 a1, _QWORD *a2, int a3, __int64 a4)
{
  __int64 v7; // r13
  __int64 v8; // r15
  char *v9; // r15
  __int64 v10; // rdi
  __int64 v11; // r14
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // rax
  _QWORD *v15; // r13
  __int64 v16; // rdi
  unsigned __int64 *v17; // r15
  __int64 v18; // rax
  unsigned __int64 v19; // rcx
  __int64 v20; // rsi
  _QWORD *v21; // rdx
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r14
  __int64 v27; // rax
  _QWORD *v28; // rdi
  unsigned int v29; // r14d
  int v30; // r13d
  __int64 v31; // rax
  __int64 v32; // rsi
  _BYTE *v33; // rdi
  char *v35; // [rsp+10h] [rbp-120h]
  __int64 v36; // [rsp+18h] [rbp-118h]
  __int64 v37; // [rsp+18h] [rbp-118h]
  int v38; // [rsp+20h] [rbp-110h]
  char *v39; // [rsp+28h] [rbp-108h]
  __int64 v40; // [rsp+30h] [rbp-100h] BYREF
  __int64 v41; // [rsp+38h] [rbp-F8h] BYREF
  _QWORD v42[2]; // [rsp+40h] [rbp-F0h] BYREF
  _BYTE v43[16]; // [rsp+50h] [rbp-E0h] BYREF
  __int16 v44; // [rsp+60h] [rbp-D0h]
  _BYTE *v45; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v46; // [rsp+78h] [rbp-B8h]
  _BYTE v47[176]; // [rsp+80h] [rbp-B0h] BYREF

  v7 = *(_QWORD *)(*(_QWORD *)(a4 + 72) + 16LL);
  v8 = *(_QWORD *)(v7 + 16);
  v36 = *(_QWORD *)(v8 + 16);
  v38 = sub_12A6F10(
          *(_QWORD *)(v36 + 16),
          1u,
          "unexpected 'rowcol' operand",
          "'rowcol' operand can be 0 or 1 only",
          (_DWORD *)(a4 + 36));
  v39 = sub_128F980((__int64)a2, v7);
  v9 = sub_128F980((__int64)a2, v8);
  v35 = sub_128F980((__int64)a2, v36);
  sub_12B24C0(a2[4], a3, v38, &v41, &v40);
  v10 = a2[5];
  v45 = v47;
  v46 = 0x1000000000LL;
  v11 = sub_1643360(v10);
  v12 = sub_15A0680(v11, v41, 0);
  v13 = (unsigned int)v46;
  if ( (unsigned int)v46 >= HIDWORD(v46) )
  {
    sub_16CD150(&v45, v47, 0, 8);
    v13 = (unsigned int)v46;
  }
  *(_QWORD *)&v45[8 * v13] = v12;
  LODWORD(v46) = v46 + 1;
  if ( *(_BYTE *)(v40 + 8) == 16 )
  {
    v15 = (_QWORD *)sub_12A8C20(a2, v40, v9);
    v23 = (unsigned int)v46;
    if ( (unsigned int)v46 < HIDWORD(v46) )
      goto LABEL_14;
    goto LABEL_24;
  }
  v37 = v40;
  v44 = 257;
  v14 = sub_1648A60(64, 1);
  v15 = (_QWORD *)v14;
  if ( v14 )
    sub_15F9210(v14, v37, v9, 0, 0, 0);
  v16 = a2[7];
  if ( v16 )
  {
    v17 = (unsigned __int64 *)a2[8];
    sub_157E9D0(v16 + 40, v15);
    v18 = v15[3];
    v19 = *v17;
    v15[4] = v17;
    v19 &= 0xFFFFFFFFFFFFFFF8LL;
    v15[3] = v19 | v18 & 7;
    *(_QWORD *)(v19 + 8) = v15 + 3;
    *v17 = *v17 & 7 | (unsigned __int64)(v15 + 3);
  }
  sub_164B780(v15, v43);
  v20 = a2[6];
  if ( v20 )
  {
    v42[0] = a2[6];
    sub_1623A60(v42, v20, 2);
    v21 = v15 + 6;
    if ( v15[6] )
    {
      sub_161E7C0(v15 + 6);
      v21 = v15 + 6;
    }
    v22 = v42[0];
    v15[6] = v42[0];
    if ( v22 )
      sub_1623210(v42, v22, v21);
  }
  v23 = (unsigned int)v46;
  if ( (unsigned int)v46 >= HIDWORD(v46) )
  {
LABEL_24:
    sub_16CD150(&v45, v47, 0, 8);
    v23 = (unsigned int)v46;
  }
LABEL_14:
  *(_QWORD *)&v45[8 * v23] = v15;
  v24 = (unsigned int)(v46 + 1);
  LODWORD(v46) = v24;
  if ( HIDWORD(v46) <= (unsigned int)v24 )
  {
    sub_16CD150(&v45, v47, 0, 8);
    v24 = (unsigned int)v46;
  }
  *(_QWORD *)&v45[8 * v24] = v39;
  v25 = (unsigned int)(v46 + 1);
  LODWORD(v46) = v25;
  if ( HIDWORD(v46) <= (unsigned int)v25 )
  {
    sub_16CD150(&v45, v47, 0, 8);
    v25 = (unsigned int)v46;
  }
  *(_QWORD *)&v45[8 * v25] = v35;
  LODWORD(v46) = v46 + 1;
  v26 = sub_15A0680(v11, 0, 0);
  v27 = (unsigned int)v46;
  if ( (unsigned int)v46 >= HIDWORD(v46) )
  {
    sub_16CD150(&v45, v47, 0, 8);
    v27 = (unsigned int)v46;
  }
  *(_QWORD *)&v45[8 * v27] = v26;
  v28 = (_QWORD *)a2[4];
  v30 = (int)v45;
  LODWORD(v46) = v46 + 1;
  v29 = v46;
  v42[0] = v40;
  v42[1] = *(_QWORD *)v39;
  v44 = 257;
  v31 = sub_126A190(v28, 4178, (__int64)v42, 2u);
  v32 = *(_QWORD *)(v31 + 24);
  sub_1285290(a2 + 6, v32, v31, v30, v29, (__int64)v43, 0);
  *(_QWORD *)a1 = 0;
  *(_BYTE *)(a1 + 12) &= ~1u;
  v33 = v45;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  if ( v33 != v47 )
    _libc_free(v33, v32);
  return a1;
}
