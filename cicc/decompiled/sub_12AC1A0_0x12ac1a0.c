// Function: sub_12AC1A0
// Address: 0x12ac1a0
//
__int64 __fastcall sub_12AC1A0(__int64 a1, _QWORD *a2, int a3, __int64 a4, char a5, char a6)
{
  __int64 v6; // rax
  unsigned int v8; // r12d
  __int64 v9; // r15
  __int64 v10; // r14
  __int64 v11; // r13
  char *v12; // r15
  char *v13; // r14
  char *v14; // rax
  _QWORD *v15; // rdi
  char *v16; // r13
  __int64 v17; // rax
  __int64 v18; // rsi
  unsigned int i; // r12d
  __int64 v20; // r10
  __int64 v21; // r14
  unsigned __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  _QWORD *v25; // r14
  __int64 v26; // rdi
  unsigned __int64 v27; // rsi
  __int64 v28; // rax
  __int64 v29; // rsi
  _QWORD *v30; // rdx
  __int64 v31; // rsi
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v36; // [rsp+10h] [rbp-B0h]
  char *v37; // [rsp+18h] [rbp-A8h]
  __int64 *v38; // [rsp+20h] [rbp-A0h]
  __int64 v40; // [rsp+28h] [rbp-98h]
  __int64 v42; // [rsp+30h] [rbp-90h]
  unsigned __int64 *v43; // [rsp+30h] [rbp-90h]
  char v44; // [rsp+3Ch] [rbp-84h]
  int v45; // [rsp+3Ch] [rbp-84h]
  __int64 v46; // [rsp+40h] [rbp-80h] BYREF
  __int64 v47; // [rsp+48h] [rbp-78h] BYREF
  _QWORD v48[4]; // [rsp+50h] [rbp-70h] BYREF
  _BYTE v49[16]; // [rsp+70h] [rbp-50h] BYREF
  __int16 v50; // [rsp+80h] [rbp-40h]

  v6 = (unsigned int)(a3 - 678);
  if ( (unsigned int)v6 > 0x1D )
  {
    v33 = (unsigned int)(a3 - 708);
    if ( (unsigned int)v33 > 0x17 )
    {
      v34 = (unsigned int)(a3 - 732);
      if ( (unsigned int)v34 > 0xC )
        sub_127B630("unexpected WMMA intrinsic!", 0);
      v44 = 0;
      v8 = dword_4281020[v34];
    }
    else
    {
      v44 = 0;
      v8 = dword_4281060[v33];
    }
  }
  else
  {
    v44 = 1;
    v8 = dword_42810C0[v6];
  }
  v9 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a4 + 72) + 16LL) + 16LL);
  v38 = *(__int64 **)(*(_QWORD *)(a4 + 72) + 16LL);
  v10 = *(_QWORD *)(v9 + 16);
  v11 = *(_QWORD *)(v10 + 16);
  sub_12A6F10(v11, 1u, "unexpected 'rowcol' operand", "'rowcol' operand can be 0 or 1 only", (_DWORD *)(a4 + 36));
  v37 = sub_128F980((__int64)a2, (__int64)v38);
  v12 = sub_128F980((__int64)a2, v9);
  v13 = sub_128F980((__int64)a2, v10);
  v14 = sub_128F980((__int64)a2, v11);
  v15 = (_QWORD *)a2[4];
  v16 = v14;
  v46 = *(_QWORD *)v12;
  v17 = sub_126A190(v15, v8, (__int64)&v46, 1u);
  v48[0] = v12;
  v50 = 257;
  v18 = *(_QWORD *)(v17 + 24);
  v48[2] = v16;
  v48[1] = v13;
  v36 = sub_1285290(a2 + 6, v18, v17, (int)v48, 3, (__int64)v49, 0);
  if ( !v44 )
  {
    if ( v8 <= 0xFC2 )
    {
      if ( v8 > 0xFA5 )
      {
        switch ( v8 )
        {
          case 0xFA6u:
          case 0xFA7u:
          case 0xFA8u:
          case 0xFA9u:
          case 0xFC2u:
            v45 = 2;
            goto LABEL_7;
          case 0xFAAu:
          case 0xFB2u:
          case 0xFBAu:
            goto LABEL_21;
          case 0xFAEu:
          case 0xFAFu:
          case 0xFB8u:
          case 0xFB9u:
            goto LABEL_6;
          case 0xFB0u:
          case 0xFB1u:
          case 0xFB6u:
          case 0xFB7u:
          case 0xFBEu:
          case 0xFBFu:
          case 0xFC0u:
          case 0xFC1u:
            goto LABEL_25;
          default:
            goto LABEL_28;
        }
      }
      if ( v8 <= 0xEA6 )
      {
LABEL_25:
        v45 = 1;
        goto LABEL_7;
      }
      v45 = 2;
      if ( v8 == 3751 )
        goto LABEL_7;
    }
LABEL_28:
    sub_127B630("unexpected imma_ld intrinsic!", 0);
  }
  if ( a6 != 1 || a5 )
LABEL_21:
    v45 = 8;
  else
LABEL_6:
    v45 = 4;
LABEL_7:
  for ( i = 0; i != v45; ++i )
  {
    v20 = v36;
    if ( v45 != 1 )
    {
      v50 = 257;
      LODWORD(v47) = i;
      v20 = sub_12A9E60(a2 + 6, v36, (__int64)&v47, 1, (__int64)v49);
    }
    v40 = v20;
    v50 = 257;
    v21 = a2[4] + 8LL;
    v22 = sub_8D46C0(*v38);
    v23 = sub_127A030(v21, v22, 0);
    v42 = sub_12A8800(a2 + 6, v23, v37, i, (__int64)v49);
    v50 = 257;
    v24 = sub_1648A60(64, 2);
    v25 = (_QWORD *)v24;
    if ( v24 )
      sub_15F9650(v24, v40, v42, 0, 0);
    v26 = a2[7];
    if ( v26 )
    {
      v43 = (unsigned __int64 *)a2[8];
      sub_157E9D0(v26 + 40, v25);
      v27 = *v43;
      v28 = v25[3] & 7LL;
      v25[4] = v43;
      v27 &= 0xFFFFFFFFFFFFFFF8LL;
      v25[3] = v27 | v28;
      *(_QWORD *)(v27 + 8) = v25 + 3;
      *v43 = *v43 & 7 | (unsigned __int64)(v25 + 3);
    }
    sub_164B780(v25, v49);
    v29 = a2[6];
    if ( v29 )
    {
      v47 = a2[6];
      sub_1623A60(&v47, v29, 2);
      v30 = v25 + 6;
      if ( v25[6] )
      {
        sub_161E7C0(v25 + 6);
        v30 = v25 + 6;
      }
      v31 = v47;
      v25[6] = v47;
      if ( v31 )
        sub_1623210(&v47, v31, v30);
    }
  }
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  return a1;
}
