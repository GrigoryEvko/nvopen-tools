// Function: sub_24708B0
// Address: 0x24708b0
//
__int64 __fastcall sub_24708B0(__int64 *a1, unsigned __int8 *a2, __int64 a3)
{
  int v3; // r15d
  __int64 v4; // rax
  int v5; // edx
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rbx
  int v10; // ebx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  unsigned __int64 v14; // rbx
  __int64 *v15; // r12
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rcx
  unsigned __int64 v19; // r13
  __int64 v20; // r14
  unsigned __int8 v21; // al
  __int64 v22; // rdx
  __int64 v23; // rdx
  char v24; // al
  __int64 v25; // r8
  __int64 v26; // rdx
  unsigned __int8 v27; // al
  __int64 v28; // rdx
  unsigned __int64 v29; // rax
  char v30; // al
  int v31; // eax
  __int64 v32; // rdx
  __int64 v33; // rdx
  unsigned __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // r13
  unsigned int v37; // r13d
  __int64 v38; // rsi
  int v39; // eax
  __int64 v40; // rax
  __int64 v41; // r12
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v45; // r8
  __int64 v46; // rdx
  __int64 v47; // rdx
  unsigned __int8 v48; // al
  __int64 v49; // rdx
  unsigned __int64 v50; // rax
  char v51; // al
  __int64 v52; // rdx
  __int64 v53; // rdx
  __int64 v54; // r8
  __int64 v55; // rax
  __int64 v56; // rsi
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // r8
  __int64 v61; // [rsp+0h] [rbp-90h]
  int v62; // [rsp+8h] [rbp-88h]
  int v63; // [rsp+8h] [rbp-88h]
  __int64 v64; // [rsp+8h] [rbp-88h]
  __int64 v65; // [rsp+8h] [rbp-88h]
  __int64 v66; // [rsp+8h] [rbp-88h]
  __int64 v67; // [rsp+8h] [rbp-88h]
  __int64 v68; // [rsp+8h] [rbp-88h]
  __int64 v69; // [rsp+8h] [rbp-88h]
  __int64 v70; // [rsp+8h] [rbp-88h]
  __int64 v71; // [rsp+8h] [rbp-88h]
  unsigned int v72; // [rsp+10h] [rbp-80h]
  __int64 v73; // [rsp+10h] [rbp-80h]
  unsigned __int64 v74; // [rsp+18h] [rbp-78h]
  unsigned int v75; // [rsp+20h] [rbp-70h]
  unsigned int v76; // [rsp+24h] [rbp-6Ch]
  unsigned __int8 *v78; // [rsp+30h] [rbp-60h]
  __int64 v80; // [rsp+50h] [rbp-40h] BYREF
  __int64 v81; // [rsp+58h] [rbp-38h]

  v4 = sub_B2BEC0(a1[1]);
  v5 = *a2;
  v61 = v4;
  if ( v5 == 40 )
  {
    v6 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v6 = -32;
    if ( v5 != 85 )
    {
      v6 = -96;
      if ( v5 != 34 )
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_9;
  v7 = sub_BD2BC0((__int64)a2);
  v9 = v7 + v8;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v9 >> 4) )
      goto LABEL_84;
  }
  else if ( (unsigned int)((v9 - sub_BD2BC0((__int64)a2)) >> 4) )
  {
    if ( (a2[7] & 0x80u) != 0 )
    {
      v10 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
      if ( (a2[7] & 0x80u) == 0 )
        BUG();
      v11 = sub_BD2BC0((__int64)a2);
      v6 -= 32LL * (unsigned int)(*(_DWORD *)(v11 + v12 - 4) - v10);
      goto LABEL_9;
    }
LABEL_84:
    BUG();
  }
LABEL_9:
  v78 = &a2[v6];
  v13 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
  if ( &a2[-v13] == &a2[v6] )
  {
    v41 = 0;
    goto LABEL_44;
  }
  v14 = 0;
  v15 = (__int64 *)&a2[-v13];
  v72 = 192;
  v75 = 64;
  v76 = 0;
  do
  {
    while ( 1 )
    {
      v19 = (unsigned int)(*(_DWORD *)(*((_QWORD *)a2 + 10) + 12LL) - 1);
      v20 = *(_QWORD *)(*v15 + 8);
      if ( (*(_BYTE *)(v20 + 8) & 0xFD) == 0xC )
      {
        v80 = sub_BCAE30(*(_QWORD *)(*v15 + 8));
        v81 = v16;
        if ( (unsigned __int64)sub_CA1930(&v80) <= 0x40 )
        {
          v17 = 8;
          LODWORD(v18) = 1;
          goto LABEL_13;
        }
      }
      v21 = *(_BYTE *)(v20 + 8);
      if ( v21 > 3u && v21 != 5 && (v21 & 0xFD) != 4 )
        break;
      v80 = sub_BCAE30(v20);
      v81 = v22;
      if ( (unsigned __int64)sub_CA1930(&v80) > 0x80 )
        break;
      v23 = 16;
      LODWORD(v18) = 1;
LABEL_21:
      if ( v23 + (unsigned __int64)v75 > 0xC0 )
        goto LABEL_39;
      v63 = v18;
      v74 = sub_2464620((__int64)a1, (unsigned int **)a3, v75);
      v75 += 16 * v63;
LABEL_15:
      if ( v19 <= v14 )
        goto LABEL_42;
LABEL_16:
      ++v14;
      v15 += 4;
      if ( v78 == (unsigned __int8 *)v15 )
        goto LABEL_43;
    }
    v24 = *(_BYTE *)(v20 + 8);
    if ( v24 == 16 )
    {
      v45 = **(_QWORD **)(v20 + 16);
      if ( (*(_BYTE *)(v45 + 8) & 0xFD) == 0xC )
      {
        v67 = **(_QWORD **)(v20 + 16);
        v80 = sub_BCAE30(v67);
        v81 = v46;
        if ( (unsigned __int64)sub_CA1930(&v80) <= 0x40 )
        {
          v47 = 1;
          v31 = 0;
LABEL_55:
          if ( (unsigned int)*(unsigned __int8 *)(v20 + 8) - 17 <= 1 )
            v20 = **(_QWORD **)(v20 + 16);
          v18 = v47 * *(_QWORD *)(v20 + 32);
          goto LABEL_36;
        }
        v45 = v67;
      }
      v48 = *(_BYTE *)(v45 + 8);
      if ( (v48 <= 3u || v48 == 5 || (v48 & 0xFD) == 4)
        && (v68 = v45, v80 = sub_BCAE30(v45), v81 = v49, v50 = sub_CA1930(&v80), v45 = v68, v50 <= 0x80) )
      {
        v47 = 1;
        v31 = 1;
      }
      else
      {
        v51 = *(_BYTE *)(v45 + 8);
        if ( v51 == 16 )
        {
          v71 = v45;
          v31 = sub_2462210((__int64)a1, **(_QWORD **)(v45 + 16));
          v60 = v71;
          if ( (unsigned int)*(unsigned __int8 *)(v71 + 8) - 17 <= 1 )
            v60 = **(_QWORD **)(v71 + 16);
          v47 = *(_QWORD *)(v60 + 32) * v59;
        }
        else if ( v51 == 17 )
        {
          v69 = v45;
          v31 = sub_2462210((__int64)a1, **(_QWORD **)(v45 + 16));
          v47 = v52 * *(unsigned int *)(v69 + 32);
        }
        else
        {
          v47 = 0;
          v31 = 2;
        }
      }
      goto LABEL_55;
    }
    if ( v24 != 17 )
      goto LABEL_39;
    v25 = **(_QWORD **)(v20 + 16);
    if ( (*(_BYTE *)(v25 + 8) & 0xFD) == 0xC )
    {
      v64 = **(_QWORD **)(v20 + 16);
      v80 = sub_BCAE30(v64);
      v81 = v26;
      if ( (unsigned __int64)sub_CA1930(&v80) <= 0x40 )
      {
        v18 = *(unsigned int *)(v20 + 32);
        goto LABEL_46;
      }
      v25 = v64;
    }
    v27 = *(_BYTE *)(v25 + 8);
    if ( v27 <= 3u || v27 == 5 || (v27 & 0xFD) == 4 )
    {
      v65 = v25;
      v80 = sub_BCAE30(v25);
      v81 = v28;
      v29 = sub_CA1930(&v80);
      v25 = v65;
      if ( v29 <= 0x80 )
      {
        v18 = *(unsigned int *)(v20 + 32);
LABEL_68:
        v23 = 16 * v18;
        goto LABEL_21;
      }
    }
    v30 = *(_BYTE *)(v25 + 8);
    if ( v30 == 16 )
    {
      v70 = v25;
      v31 = sub_2462210((__int64)a1, **(_QWORD **)(v25 + 16));
      v54 = v70;
      if ( (unsigned int)*(unsigned __int8 *)(v70 + 8) - 17 <= 1 )
        v54 = **(_QWORD **)(v70 + 16);
      v33 = *(_QWORD *)(v54 + 32) * v53;
    }
    else
    {
      if ( v30 != 17 )
        goto LABEL_39;
      v66 = v25;
      v31 = sub_2462210((__int64)a1, **(_QWORD **)(v25 + 16));
      v33 = v32 * *(unsigned int *)(v66 + 32);
    }
    v18 = v33 * *(unsigned int *)(v20 + 32);
LABEL_36:
    if ( v31 )
    {
      if ( v31 != 1 )
      {
        if ( v31 == 2 )
          goto LABEL_39;
        goto LABEL_15;
      }
      goto LABEL_68;
    }
LABEL_46:
    v17 = 8 * v18;
LABEL_13:
    if ( v17 + (unsigned __int64)v76 <= 0x40 )
    {
      v62 = v18;
      v74 = sub_2464620((__int64)a1, (unsigned int **)a3, v76);
      v76 += 8 * v62;
      goto LABEL_15;
    }
LABEL_39:
    if ( v19 > v14 )
      goto LABEL_16;
    v34 = sub_BDB740(v61, *(_QWORD *)(*v15 + 8));
    v81 = v35;
    v80 = v34;
    v36 = sub_CA1930(&v80);
    v74 = sub_2464620((__int64)a1, (unsigned int **)a3, v72);
    v37 = v72 + 8 * ((v36 != 0) + (unsigned int)((v36 - (unsigned __int64)(v36 != 0)) >> 3));
    if ( v37 > 0x320 )
    {
      if ( v72 <= 0x31F )
      {
        v55 = sub_BCB2D0(*(_QWORD **)(a3 + 72));
        v56 = 800 - v72;
        v73 = sub_ACD640(v55, v56, 1u);
        v57 = sub_BCB2B0(*(_QWORD **)(a3 + 72));
        v58 = sub_AD6530(v57, v56);
        sub_B34240(a3, v74, v58, v73, 0x103u, 0, 0, 0, 0);
      }
      v72 = v37;
      goto LABEL_16;
    }
    v72 = v37;
LABEL_42:
    LOBYTE(v3) = byte_4FE8EA8;
    v38 = *v15;
    ++v14;
    v15 += 4;
    v39 = v3;
    BYTE1(v39) = 1;
    v3 = v39;
    v40 = sub_246F3F0(a1[3], v38);
    sub_2463EC0((__int64 *)a3, v40, v74, v3, 0);
  }
  while ( v78 != (unsigned __int8 *)v15 );
LABEL_43:
  v41 = v72 - 192;
LABEL_44:
  v42 = sub_BCB2E0(*(_QWORD **)(a3 + 72));
  v43 = sub_ACD640(v42, v41, 0);
  return sub_2463EC0((__int64 *)a3, v43, *(_QWORD *)(a1[2] + 152), 0, 0);
}
