// Function: sub_2470080
// Address: 0x2470080
//
__int64 __fastcall sub_2470080(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  __int64 v6; // rax
  int v7; // edx
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r12
  int v12; // r12d
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  unsigned __int64 v16; // r14
  __int64 v17; // rsi
  unsigned __int64 v18; // rax
  __int64 v19; // rdx
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // r11
  unsigned int v22; // r12d
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned __int64 v27; // r12
  __int64 v28; // rsi
  unsigned __int8 v29; // dl
  unsigned __int8 v30; // al
  unsigned __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  int v34; // r12d
  unsigned __int64 v35; // r10
  __int64 v36; // r12
  unsigned __int16 v37; // ax
  __int64 v38; // rdx
  __int64 *v39; // rdi
  char *v40; // rax
  unsigned __int16 v41; // ax
  __int64 v42; // rax
  _BYTE *v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rax
  unsigned int v47; // esi
  unsigned int v48; // edx
  unsigned int v49; // ecx
  char v50; // r12
  __int64 v51; // rax
  __int64 v52; // rax
  unsigned int v53; // edx
  unsigned int v54; // ecx
  __int64 v55; // rax
  __int64 v56; // r12
  __int64 v57; // rax
  __int64 v58; // rax
  unsigned __int64 v60; // rax
  unsigned __int64 v61; // rax
  __int64 v62; // rdx
  unsigned __int64 v63; // rax
  unsigned __int16 v64; // [rsp+4h] [rbp-9Ch]
  unsigned __int16 v65; // [rsp+6h] [rbp-9Ah]
  __int64 v66; // [rsp+8h] [rbp-98h]
  __int64 v67; // [rsp+10h] [rbp-90h]
  __int64 v68; // [rsp+18h] [rbp-88h]
  unsigned __int16 v69; // [rsp+22h] [rbp-7Eh]
  unsigned int v70; // [rsp+24h] [rbp-7Ch]
  unsigned int v71; // [rsp+28h] [rbp-78h]
  unsigned __int16 v72; // [rsp+2Ch] [rbp-74h]
  unsigned __int16 v73; // [rsp+2Eh] [rbp-72h]
  __int64 v74; // [rsp+30h] [rbp-70h]
  __int64 v75; // [rsp+38h] [rbp-68h]
  __int64 v76; // [rsp+38h] [rbp-68h]
  __int64 v77; // [rsp+38h] [rbp-68h]
  unsigned __int64 v78; // [rsp+38h] [rbp-68h]
  unsigned __int64 v79; // [rsp+38h] [rbp-68h]
  unsigned __int64 v80; // [rsp+38h] [rbp-68h]
  __int64 v81; // [rsp+40h] [rbp-60h]
  __int64 v82; // [rsp+40h] [rbp-60h]
  __int64 v83; // [rsp+40h] [rbp-60h]
  __int64 *v84; // [rsp+48h] [rbp-58h]
  unsigned int v85; // [rsp+50h] [rbp-50h]
  __int64 v86; // [rsp+50h] [rbp-50h]
  __int64 v87; // [rsp+50h] [rbp-50h]
  __int64 *v88; // [rsp+58h] [rbp-48h]
  __int64 v89; // [rsp+60h] [rbp-40h] BYREF
  __int64 v90; // [rsp+68h] [rbp-38h]

  v85 = *(_DWORD *)(a1 + 180);
  v6 = sub_B2BEC0(*(_QWORD *)(a1 + 8));
  v7 = *a2;
  v74 = v6;
  if ( v7 == 40 )
  {
    v8 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v8 = -32;
    if ( v7 != 85 )
    {
      v8 = -96;
      if ( v7 != 34 )
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) != 0 )
  {
    v9 = sub_BD2BC0((__int64)a2);
    v11 = v9 + v10;
    if ( (a2[7] & 0x80u) == 0 )
    {
      if ( !(unsigned int)(v11 >> 4) )
        goto LABEL_9;
    }
    else
    {
      if ( !(unsigned int)((v11 - sub_BD2BC0((__int64)a2)) >> 4) )
        goto LABEL_9;
      if ( (a2[7] & 0x80u) != 0 )
      {
        v12 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
        if ( (a2[7] & 0x80u) == 0 )
          BUG();
        v13 = sub_BD2BC0((__int64)a2);
        v8 -= 32LL * (unsigned int)(*(_DWORD *)(v13 + v14 - 4) - v12);
        goto LABEL_9;
      }
    }
    BUG();
  }
LABEL_9:
  v84 = (__int64 *)&a2[v8];
  v15 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
  v88 = (__int64 *)&a2[-v15];
  if ( &a2[-v15] != &a2[v8] )
  {
    v70 = 48;
    v16 = 0;
    v71 = 0;
    do
    {
      v27 = (unsigned int)(*(_DWORD *)(*((_QWORD *)a2 + 10) + 12LL) - 1);
      if ( (unsigned __int8)sub_B49B80((__int64)a2, v16, 81) )
      {
        if ( v27 <= v16 )
        {
          v17 = sub_A748A0((_QWORD *)a2 + 9, v16);
          if ( !v17 )
          {
            v55 = *((_QWORD *)a2 - 4);
            if ( v55 )
            {
              if ( !*(_BYTE *)v55 && *(_QWORD *)(v55 + 24) == *((_QWORD *)a2 + 10) )
              {
                v89 = *(_QWORD *)(v55 + 120);
                v17 = sub_A748A0(&v89, v16);
              }
            }
          }
          v18 = sub_BDB740(v74, v17);
          v90 = v19;
          v89 = v18;
          v75 = sub_CA1930(&v89);
          v20 = sub_2464620(a1, (unsigned int **)a3, v85);
          v21 = 0;
          v81 = v20;
          if ( *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4LL) )
            v21 = sub_24648B0(a1, (unsigned int **)a3, v85);
          v22 = v85 + 8 * ((v75 != 0) + (unsigned int)((v75 - (unsigned __int64)(v75 != 0)) >> 3));
          if ( v22 <= 0x320 )
          {
            LOBYTE(v41) = byte_4FE8EA8;
            v67 = v21;
            HIBYTE(v41) = 1;
            v87 = *(_QWORD *)(a1 + 24);
            v73 = v41;
            v42 = sub_BCB2B0(*(_QWORD **)(a3 + 72));
            v43 = sub_2466120(v87, *v88, (unsigned int **)a3, v42, v73, 0);
            v66 = v44;
            LOBYTE(v87) = byte_4FE8EA8;
            v68 = (__int64)v43;
            v45 = sub_BCB2E0(*(_QWORD **)(a3 + 72));
            v46 = sub_ACD640(v45, v75, 0);
            v47 = v72;
            v48 = v69;
            LOBYTE(v47) = v87;
            LOBYTE(v48) = v87;
            v49 = v47;
            BYTE1(v48) = 1;
            BYTE1(v49) = 1;
            v69 = v48;
            v72 = v49;
            sub_B343C0(a3, 0xEEu, v81, v48, v68, v49, v46, 0, 0, 0, 0, 0);
            v85 = v22;
            if ( *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4LL) )
            {
              v50 = byte_4FE8EA8;
              v51 = sub_BCB2E0(*(_QWORD **)(a3 + 72));
              v52 = sub_ACD640(v51, v75, 0);
              v53 = v65;
              v54 = v64;
              LOBYTE(v53) = v50;
              LOBYTE(v54) = v50;
              BYTE1(v53) = 1;
              BYTE1(v54) = 1;
              v65 = v53;
              v64 = v54;
              sub_B343C0(a3, 0xEEu, v67, v54, v66, v53, v52, 0, 0, 0, 0, 0);
            }
            goto LABEL_18;
          }
          if ( v85 > 0x31F )
            goto LABEL_54;
LABEL_17:
          v23 = sub_BCB2D0(*(_QWORD **)(a3 + 72));
          v24 = 800 - v85;
          v86 = sub_ACD640(v23, v24, 1u);
          v25 = sub_BCB2B0(*(_QWORD **)(a3 + 72));
          v26 = sub_AD6530(v25, v24);
          sub_B34240(a3, v81, v26, v86, 0x103u, 0, 0, 0, 0);
          v85 = v22;
        }
      }
      else
      {
        v28 = *(_QWORD *)(*v88 + 8);
        v29 = *(_BYTE *)(v28 + 8);
        if ( v29 == 4 )
          goto LABEL_25;
        v30 = *(_BYTE *)(v28 + 8);
        if ( (unsigned int)v29 - 17 <= 1 )
          v30 = *(_BYTE *)(**(_QWORD **)(v28 + 16) + 8LL);
        if ( v30 <= 3u || v30 == 5 || (v30 & 0xFD) == 4 )
        {
          if ( *(_DWORD *)(a1 + 180) <= v70 )
            goto LABEL_25;
          v82 = 0;
          v35 = sub_2464620(a1, (unsigned int **)a3, v70);
          if ( *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4LL) )
          {
            v79 = v35;
            v61 = sub_24648B0(a1, (unsigned int **)a3, v70);
            v35 = v79;
            v82 = v61;
          }
          v70 += 16;
        }
        else
        {
          if ( v29 == 12
            && (v83 = *(_QWORD *)(*v88 + 8),
                v89 = sub_BCAE30(v83),
                v90 = v62,
                v28 = v83,
                (unsigned __int64)sub_CA1930(&v89) <= 0x40) )
          {
            if ( v71 > 0x2F )
              goto LABEL_25;
          }
          else if ( *(_BYTE *)(v28 + 8) != 14 || v71 > 0x2F )
          {
LABEL_25:
            if ( v27 > v16 )
              goto LABEL_18;
            v31 = sub_BDB740(v74, *(_QWORD *)(*v88 + 8));
            v90 = v32;
            v89 = v31;
            v33 = sub_CA1930(&v89);
            v34 = 8 * (((v33 - (unsigned __int64)(v33 != 0)) >> 3) + (v33 != 0));
            v82 = 0;
            v35 = sub_2464620(a1, (unsigned int **)a3, v85);
            if ( *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4LL) )
            {
              v80 = v35;
              v63 = sub_24648B0(a1, (unsigned int **)a3, v85);
              v22 = v85 + v34;
              v35 = v80;
              v82 = v63;
              if ( v22 <= 0x320 )
              {
LABEL_28:
                v85 = v22;
LABEL_33:
                v76 = v35;
                v36 = sub_246F3F0(*(_QWORD *)(a1 + 24), *v88);
                LOBYTE(v37) = byte_4FE8EA8;
                HIBYTE(v37) = 1;
                sub_2463EC0((__int64 *)a3, v36, v76, v37, 0);
                if ( *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4LL) )
                {
                  v77 = sub_246EE10(*(_QWORD *)(a1 + 24), *v88);
                  v89 = sub_9C6480(v74, *(_QWORD *)(v36 + 8));
                  v39 = *(__int64 **)(a1 + 24);
                  v90 = v38;
                  v40 = &byte_4FE8EA8;
                  if ( (unsigned __int8)byte_4FE8EA8 < (unsigned __int8)byte_4FE8EA9 )
                    v40 = &byte_4FE8EA9;
                  sub_24677C0(v39, a3, v77, v82, v89, v38, *v40);
                }
                goto LABEL_18;
              }
            }
            else
            {
              v22 = v85 + v34;
              if ( v22 <= 0x320 )
                goto LABEL_28;
            }
            v81 = v35;
            if ( v85 > 0x31F )
            {
LABEL_54:
              v85 = v22;
              goto LABEL_18;
            }
            goto LABEL_17;
          }
          v35 = sub_2464620(a1, (unsigned int **)a3, v71);
          if ( *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4LL) )
          {
            v78 = v35;
            v60 = sub_24648B0(a1, (unsigned int **)a3, v71);
            v35 = v78;
            v82 = v60;
          }
          else
          {
            v82 = 0;
          }
          v71 += 8;
        }
        if ( v27 <= v16 )
          goto LABEL_33;
      }
LABEL_18:
      v88 += 4;
      ++v16;
    }
    while ( v88 != v84 );
  }
  v56 = v85 - *(_DWORD *)(a1 + 180);
  v57 = sub_BCB2E0(*(_QWORD **)(a3 + 72));
  v58 = sub_ACD640(v57, v56, 0);
  return sub_2463EC0((__int64 *)a3, v58, *(_QWORD *)(*(_QWORD *)(a1 + 16) + 152LL), 0, 0);
}
