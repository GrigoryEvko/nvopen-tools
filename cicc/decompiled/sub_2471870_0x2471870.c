// Function: sub_2471870
// Address: 0x2471870
//
void __fastcall sub_2471870(__int64 *a1, unsigned __int8 *a2, __int64 a3)
{
  __int64 v4; // rbx
  _BYTE *v5; // rsi
  __int64 v6; // rdx
  int v7; // ecx
  int v8; // r15d
  __int64 v9; // rax
  int v10; // edx
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rbx
  int v15; // ebx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 *v19; // r14
  int v20; // r12d
  unsigned __int64 v21; // rbx
  __int64 v22; // rsi
  unsigned __int64 v23; // rax
  __int64 v24; // rdx
  unsigned __int16 v25; // ax
  char v26; // cl
  __int64 v27; // rax
  __int64 v28; // r15
  int v29; // r15d
  unsigned __int16 v30; // ax
  __int64 v31; // r12
  __int64 v32; // rax
  _BYTE *v33; // r12
  __int64 v34; // rax
  __int64 v35; // rax
  unsigned int v36; // esi
  unsigned int v37; // edx
  unsigned int v38; // ecx
  int v39; // eax
  unsigned __int64 v40; // rax
  __int64 v41; // rdx
  unsigned __int64 v42; // r15
  __int64 v43; // rdx
  int v44; // eax
  unsigned __int64 v45; // rcx
  unsigned __int64 v46; // rdx
  __int64 v47; // rcx
  int v48; // eax
  int v49; // r12d
  int v50; // eax
  unsigned __int16 v51; // ax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 *v54; // rax
  __int64 v55; // rdx
  unsigned __int64 v56; // rax
  unsigned __int64 v57; // rcx
  __int64 v58; // rsi
  __int64 v59; // rax
  unsigned __int64 v60; // rax
  int v61; // rdx^4
  int v62; // edx
  char v63; // [rsp+Bh] [rbp-D5h]
  unsigned __int16 v64; // [rsp+Ch] [rbp-D4h]
  unsigned __int16 v65; // [rsp+Eh] [rbp-D2h]
  unsigned __int64 v66; // [rsp+18h] [rbp-C8h]
  __int64 *v68; // [rsp+30h] [rbp-B0h]
  __int64 v69; // [rsp+40h] [rbp-A0h]
  unsigned __int64 v70; // [rsp+40h] [rbp-A0h]
  _BYTE *v71; // [rsp+48h] [rbp-98h]
  unsigned __int16 v72; // [rsp+50h] [rbp-90h]
  unsigned __int16 v73; // [rsp+52h] [rbp-8Eh]
  int v74; // [rsp+54h] [rbp-8Ch]
  unsigned __int64 v75; // [rsp+58h] [rbp-88h]
  unsigned __int64 v76; // [rsp+60h] [rbp-80h] BYREF
  __int64 v77; // [rsp+68h] [rbp-78h]
  __int64 v78[2]; // [rsp+70h] [rbp-70h] BYREF
  _QWORD v79[2]; // [rsp+80h] [rbp-60h] BYREF
  __int64 v80; // [rsp+90h] [rbp-50h]
  __int64 v81; // [rsp+98h] [rbp-48h]
  __int64 v82; // [rsp+A0h] [rbp-40h]

  v4 = *(_QWORD *)(a1[1] + 40);
  v5 = *(_BYTE **)(v4 + 232);
  v6 = *(_QWORD *)(v4 + 240);
  v78[0] = (__int64)v79;
  sub_2462160(v78, v5, (__int64)&v5[v6]);
  v7 = *(_DWORD *)(v4 + 276);
  v80 = *(_QWORD *)(v4 + 264);
  v81 = *(_QWORD *)(v4 + 272);
  v82 = *(_QWORD *)(v4 + 280);
  if ( (_DWORD)v80 != 24 )
  {
    v8 = 8;
    if ( (_DWORD)v80 != 25 )
      goto LABEL_3;
    goto LABEL_55;
  }
  if ( v7 == 3 )
  {
    if ( (unsigned int)sub_CC78E0((__int64)v78) > 0xC )
      goto LABEL_61;
    v60 = sub_CC78E0((__int64)v78);
    if ( !((v62 | HIDWORD(v60)) & 0x7FFFFFFF | (unsigned int)v60 | v61 & 0x7FFFFFFF) )
      goto LABEL_61;
    v7 = HIDWORD(v81);
  }
  if ( v7 != 11 && v7 != 39 && (_DWORD)v82 != 49 && (unsigned int)(v82 - 18) > 7 )
  {
LABEL_55:
    v8 = 48;
    goto LABEL_3;
  }
LABEL_61:
  v8 = 32;
LABEL_3:
  v9 = sub_B2BEC0(a1[1]);
  v10 = *a2;
  v71 = (_BYTE *)v9;
  if ( v10 == 40 )
  {
    v11 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v11 = -32;
    if ( v10 != 85 )
    {
      v11 = -96;
      if ( v10 != 34 )
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) != 0 )
  {
    v12 = sub_BD2BC0((__int64)a2);
    v14 = v12 + v13;
    if ( (a2[7] & 0x80u) == 0 )
    {
      if ( !(unsigned int)(v14 >> 4) )
        goto LABEL_11;
    }
    else
    {
      if ( !(unsigned int)((v14 - sub_BD2BC0((__int64)a2)) >> 4) )
        goto LABEL_11;
      if ( (a2[7] & 0x80u) != 0 )
      {
        v15 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
        if ( (a2[7] & 0x80u) == 0 )
          BUG();
        v16 = sub_BD2BC0((__int64)a2);
        v11 -= 32LL * (unsigned int)(*(_DWORD *)(v16 + v17 - 4) - v15);
        goto LABEL_11;
      }
    }
    BUG();
  }
LABEL_11:
  v68 = (__int64 *)&a2[v11];
  v18 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
  v19 = (__int64 *)&a2[-v18];
  if ( &a2[-v18] != &a2[v11] )
  {
    v20 = v8;
    v21 = 0;
    v74 = v8;
    while ( 1 )
    {
      v75 = (unsigned int)(*(_DWORD *)(*((_QWORD *)a2 + 10) + 12LL) - 1);
      if ( (unsigned __int8)sub_B49B80((__int64)a2, v21, 81) )
      {
        v22 = sub_A748A0((_QWORD *)a2 + 9, v21);
        if ( !v22 )
        {
          v53 = *((_QWORD *)a2 - 4);
          if ( v53 )
          {
            if ( !*(_BYTE *)v53 && *(_QWORD *)(v53 + 24) == *((_QWORD *)a2 + 10) )
            {
              v76 = *(_QWORD *)(v53 + 120);
              v22 = sub_A748A0(&v76, v21);
            }
          }
        }
        v23 = sub_BDB740((__int64)v71, v22);
        v77 = v24;
        v76 = v23;
        v69 = sub_CA1930(&v76);
        v25 = sub_A74840((_QWORD *)a2 + 9, v21);
        v26 = v25;
        if ( HIBYTE(v25) && (v27 = 1LL << v25, (unsigned __int64)(1LL << v26) > 7) )
        {
          v28 = -v27;
        }
        else
        {
          LODWORD(v28) = -8;
          LODWORD(v27) = 8;
        }
        v29 = (v27 + v20 - 1) & v28;
        if ( v75 <= v21 && (unsigned int)(v29 - v74 + v69) <= 0x320 )
        {
          v66 = sub_2464620((__int64)a1, (unsigned int **)a3, v29 - v74);
          if ( v66 )
          {
            LOBYTE(v30) = byte_4FE8EA8;
            v31 = a1[3];
            HIBYTE(v30) = 1;
            v72 = v30;
            v32 = sub_BCB2B0(*(_QWORD **)(a3 + 72));
            v33 = sub_2466120(v31, *v19, (unsigned int **)a3, v32, v72, 0);
            v63 = byte_4FE8EA8;
            v34 = sub_BCB2E0(*(_QWORD **)(a3 + 72));
            v35 = sub_ACD640(v34, v69, 0);
            v36 = v65;
            v37 = v64;
            LOBYTE(v36) = v63;
            LOBYTE(v37) = v63;
            v38 = v36;
            BYTE1(v37) = 1;
            BYTE1(v38) = 1;
            v64 = v37;
            v65 = v38;
            sub_B343C0(a3, 0xEEu, v66, v37, (__int64)v33, v38, v35, 0, 0, 0, 0, 0);
          }
        }
        v20 = v29 + ((v69 + 7) & 0xFFFFFFF8);
        goto LABEL_22;
      }
      v40 = sub_BDB740((__int64)v71, *(_QWORD *)(*v19 + 8));
      v77 = v41;
      v76 = v40;
      v42 = sub_CA1930(&v76);
      v43 = *(_QWORD *)(*v19 + 8);
      v44 = *(unsigned __int8 *)(v43 + 8);
      if ( (_BYTE)v44 == 16 )
      {
        v54 = *(__int64 **)(v43 + 16);
        if ( *(_BYTE *)(*v54 + 8) == 6
          || (v76 = sub_BDB740((__int64)v71, *v54), v77 = v55, (v56 = sub_CA1930(&v76)) == 0)
          || (_BitScanReverse64(&v57, v56), v46 = 0x8000000000000000LL >> ((unsigned __int8)v57 ^ 0x3Fu), v46 <= 7) )
        {
LABEL_30:
          LODWORD(v47) = -8;
          LODWORD(v46) = 8;
          goto LABEL_31;
        }
      }
      else
      {
        if ( (unsigned int)(v44 - 17) > 1 )
          goto LABEL_30;
        if ( !v42 )
          goto LABEL_30;
        _BitScanReverse64(&v45, v42);
        v46 = 0x8000000000000000LL >> ((unsigned __int8)v45 ^ 0x3Fu);
        if ( v46 <= 7 )
          goto LABEL_30;
      }
      v47 = -(__int64)v46;
LABEL_31:
      v48 = v47 & (v46 + v20 - 1);
      v49 = v48;
      if ( *v71 )
      {
        v50 = v48 - v42 + 8;
        if ( v42 < 8 )
          v49 = v50;
      }
      if ( v75 <= v21 && (unsigned int)(v49 - v74 + v42) <= 0x320 )
      {
        v70 = sub_2464620((__int64)a1, (unsigned int **)a3, v49 - v74);
        if ( v70 )
        {
          LOBYTE(v51) = byte_4FE8EA8;
          HIBYTE(v51) = 1;
          v73 = v51;
          v52 = sub_246F3F0(a1[3], *v19);
          sub_2463EC0((__int64 *)a3, v52, v70, v73, 0);
        }
      }
      v20 = (v42 + v49 + 7) & 0xFFFFFFF8;
LABEL_22:
      v39 = v74;
      if ( v75 > v21 )
        v39 = v20;
      ++v21;
      v19 += 4;
      v74 = v39;
      if ( v19 == v68 )
      {
        v58 = (unsigned int)(v20 - v39);
        goto LABEL_50;
      }
    }
  }
  v58 = 0;
LABEL_50:
  v59 = sub_AD64C0(*(_QWORD *)(a1[2] + 80), v58, 0);
  sub_2463EC0((__int64 *)a3, v59, *(_QWORD *)(a1[2] + 152), 0, 0);
  if ( (_QWORD *)v78[0] != v79 )
    j_j___libc_free_0(v78[0]);
}
