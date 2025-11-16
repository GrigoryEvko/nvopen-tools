// Function: sub_33AC8F0
// Address: 0x33ac8f0
//
void __fastcall sub_33AC8F0(__int64 a1, unsigned __int8 *a2, int a3)
{
  __int64 v6; // r15
  __int64 v7; // rdx
  int v8; // edx
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  unsigned int v15; // edx
  int v16; // edx
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 *v19; // r14
  __int64 v20; // rax
  int v21; // eax
  int v22; // edx
  int v23; // r14d
  char v24; // al
  int v25; // r9d
  int v26; // r8d
  char v27; // al
  int v28; // edx
  int v29; // edx
  int v30; // edx
  int v31; // edx
  __int64 v32; // r14
  int v33; // edx
  int v34; // ebx
  _QWORD *v35; // rax
  __int64 v36; // rsi
  int v37; // edx
  int v38; // edx
  int v39; // edx
  int v40; // edx
  int v41; // edx
  int v42; // edx
  int v43; // edx
  int v44; // edx
  int v45; // edx
  int v46; // edx
  int v47; // edx
  int v48; // edx
  __int64 v49; // rbx
  __int128 v50; // rax
  int v51; // edx
  __int64 v52; // rdi
  __int128 v53; // rax
  int v54; // edx
  int v55; // edx
  int v56; // edx
  __int64 v57; // [rsp-10h] [rbp-A0h]
  __int128 v58; // [rsp+0h] [rbp-90h]
  int v59; // [rsp+14h] [rbp-7Ch]
  __int64 v60; // [rsp+18h] [rbp-78h]
  int v61; // [rsp+18h] [rbp-78h]
  int v62; // [rsp+18h] [rbp-78h]
  int v63; // [rsp+18h] [rbp-78h]
  __int128 v64; // [rsp+20h] [rbp-70h]
  unsigned __int8 *v65; // [rsp+48h] [rbp-48h] BYREF
  __int64 v66; // [rsp+50h] [rbp-40h] BYREF
  int v67; // [rsp+58h] [rbp-38h]

  v6 = *(_QWORD *)(*(_QWORD *)(a1 + 864) + 16LL);
  *(_QWORD *)&v58 = sub_338B750(a1, *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)]);
  *((_QWORD *)&v58 + 1) = v7;
  v64 = 0u;
  v8 = *a2;
  if ( v8 == 40 )
  {
    v9 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v9 = 0;
    if ( v8 != 85 )
    {
      v9 = 64;
      if ( v8 != 34 )
LABEL_64:
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_10;
  v10 = sub_BD2BC0((__int64)a2);
  v60 = v11 + v10;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v60 >> 4) )
LABEL_62:
      BUG();
LABEL_10:
    v14 = 0;
    goto LABEL_11;
  }
  if ( !(unsigned int)((v60 - sub_BD2BC0((__int64)a2)) >> 4) )
    goto LABEL_10;
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_62;
  v61 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v12 = sub_BD2BC0((__int64)a2);
  v14 = 32LL * (unsigned int)(*(_DWORD *)(v12 + v13 - 4) - v61);
LABEL_11:
  if ( (unsigned int)((32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v9 - v14) >> 5) > 1 )
  {
    *(_QWORD *)&v64 = sub_338B750(a1, *(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))]);
    *((_QWORD *)&v64 + 1) = v15;
  }
  v16 = *(_DWORD *)(a1 + 848);
  v17 = *(_QWORD *)a1;
  v66 = 0;
  v67 = v16;
  if ( v17 )
  {
    if ( &v66 != (__int64 *)(v17 + 48) )
    {
      v18 = *(_QWORD *)(v17 + 48);
      v66 = v18;
      if ( v18 )
        sub_B96E90((__int64)&v66, v18, 1);
    }
  }
  v19 = (__int64 *)*((_QWORD *)a2 + 1);
  v20 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 864) + 40LL));
  v21 = sub_2D5BAE0(v6, v20, v19, 0);
  v62 = v22;
  v23 = v21;
  v24 = sub_920620((__int64)a2);
  v26 = v62;
  if ( !v24 )
  {
    switch ( a3 )
    {
      case 387:
LABEL_53:
        v32 = sub_33FAF80(*(_QWORD *)(a1 + 864), 382, (unsigned int)&v66, v23, v62, v25);
        v34 = v47;
        goto LABEL_36;
      case 388:
LABEL_52:
        v32 = sub_33FAF80(*(_QWORD *)(a1 + 864), 384, (unsigned int)&v66, v23, v62, v25);
        v34 = v46;
        goto LABEL_36;
      case 389:
        v25 = 0;
LABEL_61:
        v32 = sub_3405C90(*(_QWORD *)(a1 + 864), 374, (unsigned int)&v66, v23, v62, v25, v58, v64);
        v34 = v56;
        goto LABEL_36;
      case 390:
        v25 = 0;
LABEL_51:
        v32 = sub_33FA050(*(_QWORD *)(a1 + 864), 378, (unsigned int)&v66, v23, v62, v25, v58, *((__int64 *)&v58 + 1));
        v34 = v45;
        goto LABEL_36;
      case 391:
        v25 = 0;
LABEL_49:
        v32 = sub_33FA050(*(_QWORD *)(a1 + 864), 380, (unsigned int)&v66, v23, v62, v25, v58, *((__int64 *)&v58 + 1));
        v34 = v44;
        goto LABEL_36;
      case 392:
        v25 = 0;
LABEL_47:
        v32 = sub_33FA050(*(_QWORD *)(a1 + 864), 379, (unsigned int)&v66, v23, v62, v25, v58, *((__int64 *)&v58 + 1));
        v34 = v43;
        goto LABEL_36;
      case 393:
        v25 = 0;
LABEL_45:
        v32 = sub_33FA050(*(_QWORD *)(a1 + 864), 381, (unsigned int)&v66, v23, v62, v25, v58, *((__int64 *)&v58 + 1));
        v34 = v42;
        goto LABEL_36;
      case 394:
        v52 = *(_QWORD *)(a1 + 864);
        v25 = 0;
LABEL_59:
        v32 = sub_3405C90(v52, 375, (unsigned int)&v66, v23, v62, v25, v58, v64);
        v34 = v55;
        goto LABEL_36;
      case 395:
LABEL_43:
        v32 = sub_33FAF80(*(_QWORD *)(a1 + 864), 383, (unsigned int)&v66, v23, v62, v25);
        v34 = v41;
        goto LABEL_36;
      case 396:
LABEL_42:
        v32 = sub_33FAF80(*(_QWORD *)(a1 + 864), 385, (unsigned int)&v66, v23, v62, v25);
        v34 = v40;
        goto LABEL_36;
      case 397:
LABEL_41:
        v32 = sub_33FAF80(*(_QWORD *)(a1 + 864), 387, (unsigned int)&v66, v23, v62, v25);
        v34 = v39;
        goto LABEL_36;
      case 398:
LABEL_40:
        v32 = sub_33FAF80(*(_QWORD *)(a1 + 864), 388, (unsigned int)&v66, v23, v62, v25);
        v34 = v38;
        goto LABEL_36;
      case 399:
LABEL_39:
        v32 = sub_33FAF80(*(_QWORD *)(a1 + 864), 389, (unsigned int)&v66, v23, v62, v25);
        v34 = v37;
        goto LABEL_36;
      case 400:
LABEL_35:
        v32 = sub_33FAF80(*(_QWORD *)(a1 + 864), 390, (unsigned int)&v66, v23, v62, v25);
        v34 = v33;
        goto LABEL_36;
      case 401:
LABEL_54:
        v32 = sub_33FAF80(*(_QWORD *)(a1 + 864), 386, (unsigned int)&v66, v23, v62, v25);
        v34 = v48;
        goto LABEL_36;
      default:
        goto LABEL_64;
    }
  }
  v27 = a2[1] >> 1;
  v25 = (16 * v27) & 0x20;
  if ( (v27 & 4) != 0 )
    v25 = (16 * v27) & 0x20 | 0x40;
  v28 = v25;
  if ( (v27 & 8) != 0 )
  {
    LOBYTE(v28) = v25 | 0x80;
    v25 = v28;
  }
  v29 = v25;
  if ( (v27 & 0x10) != 0 )
  {
    BYTE1(v29) = BYTE1(v25) | 1;
    v25 = v29;
  }
  v30 = v25;
  if ( (v27 & 0x20) != 0 )
  {
    BYTE1(v30) = BYTE1(v25) | 2;
    v25 = v30;
  }
  v31 = v25;
  if ( (v27 & 0x40) != 0 )
  {
    BYTE1(v31) = BYTE1(v25) | 4;
    v25 = v31;
  }
  if ( (a2[1] & 2) != 0 )
  {
    v25 |= 0x800u;
    switch ( a3 )
    {
      case 387:
        goto LABEL_53;
      case 388:
        goto LABEL_52;
      case 389:
        v49 = *(_QWORD *)(a1 + 864);
        v57 = v64;
        v63 = v25;
        LODWORD(v64) = v26;
        *(_QWORD *)&v50 = sub_33FA050(v49, 376, (unsigned int)&v66, v23, v26, v25, v57, *((__int64 *)&v64 + 1));
        v32 = sub_3405C90(v49, 96, (unsigned int)&v66, v23, v64, v63, v58, v50);
        v34 = v51;
        goto LABEL_36;
      case 390:
        goto LABEL_51;
      case 391:
        goto LABEL_49;
      case 392:
        goto LABEL_47;
      case 393:
        goto LABEL_45;
      case 394:
        goto LABEL_56;
      case 395:
        goto LABEL_43;
      case 396:
        goto LABEL_42;
      case 397:
        goto LABEL_41;
      case 398:
        goto LABEL_40;
      case 399:
        goto LABEL_39;
      case 400:
        goto LABEL_35;
      case 401:
        goto LABEL_54;
      default:
        goto LABEL_64;
    }
  }
  switch ( a3 )
  {
    case 387:
      goto LABEL_53;
    case 388:
      goto LABEL_52;
    case 389:
      goto LABEL_61;
    case 390:
      goto LABEL_51;
    case 391:
      goto LABEL_49;
    case 392:
      goto LABEL_47;
    case 393:
      goto LABEL_45;
    case 394:
LABEL_56:
      v52 = *(_QWORD *)(a1 + 864);
      if ( (v25 & 0x800) == 0 )
        goto LABEL_59;
      v59 = v25;
      *(_QWORD *)&v53 = sub_33FA050(v52, 377, (unsigned int)&v66, v23, v62, v25, v64, *((__int64 *)&v64 + 1));
      v32 = sub_3405C90(v52, 98, (unsigned int)&v66, v23, v62, v59, v58, v53);
      v34 = v54;
      break;
    case 395:
      goto LABEL_43;
    case 396:
      goto LABEL_42;
    case 397:
      goto LABEL_41;
    case 398:
      goto LABEL_40;
    case 399:
      goto LABEL_39;
    case 400:
      goto LABEL_35;
    case 401:
      goto LABEL_54;
    default:
      goto LABEL_64;
  }
LABEL_36:
  v65 = a2;
  v35 = sub_337DC20(a1 + 8, (__int64 *)&v65);
  *v35 = v32;
  v36 = v66;
  *((_DWORD *)v35 + 2) = v34;
  if ( v36 )
    sub_B91220((__int64)&v66, v36);
}
