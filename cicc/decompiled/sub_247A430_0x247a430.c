// Function: sub_247A430
// Address: 0x247a430
//
void __fastcall sub_247A430(__int64 *a1, unsigned __int8 *a2)
{
  __int64 v2; // rbx
  int v3; // edx
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rbx
  int v8; // ebx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  int v12; // r8d
  __int64 v13; // rdx
  _BYTE *v14; // rcx
  __int64 v15; // r8
  unsigned int v16; // r14d
  unsigned int v17; // eax
  unsigned __int64 v18; // r9
  __int64 v19; // rdx
  __int64 v20; // r9
  __int64 v21; // rax
  int v22; // edx
  __int64 v23; // r14
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rdx
  _BYTE *v29; // rax
  __int64 v30; // rax
  _BYTE *v31; // rax
  _BYTE *v32; // rax
  _BYTE *v33; // r11
  unsigned __int64 v34; // r14
  _QWORD *v35; // rax
  unsigned __int64 v36; // rax
  __int64 v37; // rax
  _BYTE *v38; // [rsp+8h] [rbp-178h]
  __int64 v39; // [rsp+10h] [rbp-170h]
  __int64 v40; // [rsp+10h] [rbp-170h]
  unsigned int v41; // [rsp+10h] [rbp-170h]
  _BYTE *v42; // [rsp+10h] [rbp-170h]
  __int64 v43; // [rsp+18h] [rbp-168h]
  int v44; // [rsp+18h] [rbp-168h]
  __int64 v45; // [rsp+18h] [rbp-168h]
  __int64 v46; // [rsp+18h] [rbp-168h]
  unsigned int v47; // [rsp+18h] [rbp-168h]
  unsigned int v48; // [rsp+18h] [rbp-168h]
  _BYTE *v49; // [rsp+18h] [rbp-168h]
  int v50; // [rsp+28h] [rbp-158h]
  __int64 v51; // [rsp+28h] [rbp-158h]
  _BYTE v52[32]; // [rsp+30h] [rbp-150h] BYREF
  __int16 v53; // [rsp+50h] [rbp-130h]
  _BYTE *v54; // [rsp+60h] [rbp-120h] BYREF
  __int64 v55; // [rsp+68h] [rbp-118h]
  _BYTE v56[32]; // [rsp+70h] [rbp-110h] BYREF
  _BYTE *v57; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v58; // [rsp+98h] [rbp-E8h]
  _BYTE v59[32]; // [rsp+A0h] [rbp-E0h] BYREF
  unsigned int *v60[24]; // [rsp+C0h] [rbp-C0h] BYREF

  v2 = *(_QWORD *)(*(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)] + 8LL);
  sub_23D0AB0((__int64)v60, (__int64)a2, 0, 0, 0);
  v3 = *a2;
  v50 = *(_DWORD *)(v2 + 32);
  if ( v3 == 40 )
  {
    v4 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v4 = 0;
    if ( v3 != 85 )
    {
      v4 = 64;
      if ( v3 != 34 )
LABEL_51:
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_10;
  v5 = sub_BD2BC0((__int64)a2);
  v7 = v5 + v6;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v7 >> 4) )
LABEL_49:
      BUG();
LABEL_10:
    v11 = 0;
    goto LABEL_11;
  }
  if ( !(unsigned int)((v7 - sub_BD2BC0((__int64)a2)) >> 4) )
    goto LABEL_10;
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_49;
  v8 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v9 = sub_BD2BC0((__int64)a2);
  v11 = 32LL * (unsigned int)(*(_DWORD *)(v9 + v10 - 4) - v8);
LABEL_11:
  v12 = *((_DWORD *)a2 + 1);
  v13 = 0;
  v14 = v56;
  v57 = v59;
  v54 = v56;
  v15 = (32LL * (v12 & 0x7FFFFFF) - 32 - v4 - v11) >> 5;
  v55 = 0x800000000LL;
  v16 = v15 * v50;
  v58 = 0x800000000LL;
  v17 = 0;
  if ( (_DWORD)v15 * v50 )
  {
    while ( 1 )
    {
      *(_DWORD *)&v14[4 * v13] = v17;
      v19 = (unsigned int)v58;
      v20 = v17 + 1;
      LODWORD(v55) = v55 + 1;
      if ( (unsigned __int64)(unsigned int)v58 + 1 > HIDWORD(v58) )
      {
        v41 = v17;
        v48 = v17 + 1;
        sub_C8D5F0((__int64)&v57, v59, (unsigned int)v58 + 1LL, 4u, v15, v20);
        v19 = (unsigned int)v58;
        v17 = v41;
        LODWORD(v20) = v48;
      }
      v17 += 2;
      *(_DWORD *)&v57[4 * v19] = v20;
      LODWORD(v58) = v58 + 1;
      if ( v16 <= v17 )
        break;
      v13 = (unsigned int)v55;
      v18 = (unsigned int)v55 + 1LL;
      if ( v18 > HIDWORD(v55) )
      {
        v47 = v17;
        sub_C8D5F0((__int64)&v54, v56, (unsigned int)v55 + 1LL, 4u, v15, v18);
        v13 = (unsigned int)v55;
        v17 = v47;
      }
      v14 = v54;
    }
  }
  v21 = sub_24723A0((__int64)a1, (__int64)a2, 0);
  v22 = *a2;
  v23 = v21;
  if ( v22 == 40 )
  {
    v51 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v51 = 0;
    if ( v22 != 85 )
    {
      v51 = 64;
      if ( v22 != 34 )
        goto LABEL_51;
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_33;
  v24 = sub_BD2BC0((__int64)a2);
  v43 = v25 + v24;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v43 >> 4) )
LABEL_47:
      BUG();
LABEL_33:
    v28 = 0;
    goto LABEL_34;
  }
  if ( !(unsigned int)((v43 - sub_BD2BC0((__int64)a2)) >> 4) )
    goto LABEL_33;
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_47;
  v44 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v26 = sub_BD2BC0((__int64)a2);
  v28 = 32LL * (unsigned int)(*(_DWORD *)(v26 + v27 - 4) - v44);
LABEL_34:
  if ( (unsigned int)((32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v51 - v28) >> 5) == 2 )
  {
    v42 = (_BYTE *)sub_24723A0((__int64)a1, (__int64)a2, 1u);
    v53 = 257;
    v37 = sub_A83CB0(v60, (_BYTE *)v23, v42, (__int64)v54, (unsigned int)v55, (__int64)v52);
    v53 = 257;
    v49 = (_BYTE *)v37;
    v32 = (_BYTE *)sub_A83CB0(v60, (_BYTE *)v23, v42, (__int64)v57, (unsigned int)v58, (__int64)v52);
    v33 = v49;
  }
  else
  {
    v53 = 257;
    v39 = (unsigned int)v55;
    v45 = (__int64)v54;
    v29 = (_BYTE *)sub_ACADE0(*(__int64 ***)(v23 + 8));
    v30 = sub_A83CB0(v60, (_BYTE *)v23, v29, v45, v39, (__int64)v52);
    v53 = 257;
    v40 = (unsigned int)v58;
    v46 = (__int64)v57;
    v38 = (_BYTE *)v30;
    v31 = (_BYTE *)sub_ACADE0(*(__int64 ***)(v23 + 8));
    v32 = (_BYTE *)sub_A83CB0(v60, (_BYTE *)v23, v31, v46, v40, (__int64)v52);
    v33 = v38;
  }
  v53 = 257;
  v34 = sub_A82480(v60, v33, v32, (__int64)v52);
  v35 = sub_2463540(a1, *((_QWORD *)a2 + 1));
  v36 = sub_2464970(a1, v60, v34, (__int64)v35, 0);
  sub_246EF60((__int64)a1, (__int64)a2, v36);
  if ( *(_DWORD *)(a1[1] + 4) )
    sub_2477350((__int64)a1, (__int64)a2);
  if ( v57 != v59 )
    _libc_free((unsigned __int64)v57);
  if ( v54 != v56 )
    _libc_free((unsigned __int64)v54);
  sub_F94A20(v60, (__int64)a2);
}
