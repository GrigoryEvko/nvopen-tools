// Function: sub_246FB00
// Address: 0x246fb00
//
__int64 __fastcall sub_246FB00(__int64 *a1, unsigned __int8 *a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  unsigned int v6; // eax
  int v7; // edx
  unsigned int v8; // r15d
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r12
  int v13; // r12d
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 *v16; // r14
  unsigned __int64 v17; // r13
  unsigned int v18; // r15d
  __int64 v19; // rsi
  unsigned __int64 v20; // rax
  __int64 v21; // rdx
  __int16 v22; // ax
  char v23; // dl
  char v24; // cl
  __int64 v25; // rax
  char v26; // r12
  __int64 v27; // rdx
  unsigned __int16 v28; // ax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  unsigned int v32; // esi
  unsigned int v33; // edx
  unsigned int v34; // ecx
  __int64 v35; // rdx
  unsigned __int64 v36; // rax
  unsigned __int64 v37; // rcx
  unsigned __int64 v38; // r12
  __int64 v39; // r9
  int v40; // r10d
  unsigned __int64 v41; // rax
  __int64 v42; // r15
  unsigned __int16 v43; // ax
  __int64 v44; // rax
  __int64 v45; // rsi
  __int64 v46; // rax
  __int64 v48; // rax
  _BYTE *v49; // [rsp+0h] [rbp-A0h]
  unsigned __int16 v50; // [rsp+Ch] [rbp-94h]
  unsigned __int16 v51; // [rsp+Eh] [rbp-92h]
  unsigned __int16 v52; // [rsp+10h] [rbp-90h]
  unsigned __int16 v53; // [rsp+12h] [rbp-8Eh]
  unsigned int v54; // [rsp+14h] [rbp-8Ch]
  __int64 v56; // [rsp+28h] [rbp-78h]
  unsigned __int64 v57; // [rsp+30h] [rbp-70h]
  __int64 *v58; // [rsp+38h] [rbp-68h]
  _BYTE *v59; // [rsp+48h] [rbp-58h]
  __int64 v60; // [rsp+50h] [rbp-50h]
  unsigned int v61; // [rsp+50h] [rbp-50h]
  int v62; // [rsp+50h] [rbp-50h]
  unsigned __int64 v63; // [rsp+58h] [rbp-48h]
  unsigned __int64 v64; // [rsp+58h] [rbp-48h]
  int v65; // [rsp+58h] [rbp-48h]
  unsigned __int64 v66; // [rsp+60h] [rbp-40h] BYREF
  __int64 v67; // [rsp+68h] [rbp-38h]

  v59 = (_BYTE *)sub_B2BEC0(a1[1]);
  v4 = sub_9208B0((__int64)v59, *(_QWORD *)(a1[2] + 80));
  v67 = v5;
  v66 = (unsigned __int64)(v4 + 7) >> 3;
  v6 = sub_CA1930(&v66);
  v7 = *a2;
  v54 = v6;
  v8 = v6;
  if ( v7 == 40 )
  {
    v9 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v9 = -32;
    if ( v7 != 85 )
    {
      v9 = -96;
      if ( v7 != 34 )
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_9;
  v10 = sub_BD2BC0((__int64)a2);
  v12 = v10 + v11;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v12 >> 4) )
      goto LABEL_50;
  }
  else if ( (unsigned int)((v12 - sub_BD2BC0((__int64)a2)) >> 4) )
  {
    if ( (a2[7] & 0x80u) != 0 )
    {
      v13 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
      if ( (a2[7] & 0x80u) == 0 )
        BUG();
      v14 = sub_BD2BC0((__int64)a2);
      v9 -= 32LL * (unsigned int)(*(_DWORD *)(v14 + v15 - 4) - v13);
      goto LABEL_9;
    }
LABEL_50:
    BUG();
  }
LABEL_9:
  v58 = (__int64 *)&a2[v9];
  v16 = (__int64 *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  if ( v16 == v58 )
  {
    v45 = 0;
    goto LABEL_34;
  }
  v17 = 0;
  v57 = v8;
  v18 = 0;
  do
  {
    while ( 1 )
    {
      v64 = (unsigned int)(*(_DWORD *)(*((_QWORD *)a2 + 10) + 12LL) - 1);
      v61 = v18 - 1;
      if ( (unsigned __int8)sub_B49B80((__int64)a2, v17, 81) )
      {
        v19 = sub_A748A0((_QWORD *)a2 + 9, v17);
        if ( !v19 )
        {
          v48 = *((_QWORD *)a2 - 4);
          if ( v48 )
          {
            if ( !*(_BYTE *)v48 && *(_QWORD *)(v48 + 24) == *((_QWORD *)a2 + 10) )
            {
              v66 = *(_QWORD *)(v48 + 120);
              v19 = sub_A748A0(&v66, v17);
            }
          }
        }
        v20 = sub_BDB740((__int64)v59, v19);
        v67 = v21;
        v66 = v20;
        v56 = sub_CA1930(&v66);
        v22 = sub_A74840((_QWORD *)a2 + 9, v17);
        v23 = HIBYTE(v22);
        v24 = v22;
        if ( v57 )
        {
          _BitScanReverse64((unsigned __int64 *)&v25, v57);
          v26 = 63 - (v25 ^ 0x3F);
          if ( !v23 )
            v24 = 63 - (v25 ^ 0x3F);
          v27 = 1LL << v24;
          if ( 1LL << v24 < v57 )
          {
            v18 = (v61 + (1 << v26)) & -(1 << v26);
            goto LABEL_17;
          }
        }
        else
        {
          if ( !HIBYTE(v22) )
          {
            v18 = 0;
            v26 = -1;
LABEL_17:
            if ( v64 <= v17 )
            {
              if ( v18 + (unsigned int)v56 <= 0x320 )
              {
                v63 = sub_2464620((__int64)a1, (unsigned int **)a3, v18);
                if ( v63 )
                {
                  LOBYTE(v28) = byte_4FE8EA8;
                  v60 = a1[3];
                  HIBYTE(v28) = 1;
                  v52 = v28;
                  v29 = sub_BCB2B0(*(_QWORD **)(a3 + 72));
                  v49 = sub_2466120(v60, *v16, (unsigned int **)a3, v29, v52, 0);
                  LOBYTE(v60) = byte_4FE8EA8;
                  v30 = sub_BCB2E0(*(_QWORD **)(a3 + 72));
                  v31 = sub_ACD640(v30, v56, 0);
                  v32 = v51;
                  v33 = v50;
                  LOBYTE(v32) = v60;
                  LOBYTE(v33) = v60;
                  v34 = v32;
                  BYTE1(v33) = 1;
                  BYTE1(v34) = 1;
                  v50 = v33;
                  v51 = v34;
                  sub_B343C0(a3, 0xEEu, v63, v33, (__int64)v49, v34, v31, 0, 0, 0, 0, 0);
                }
              }
              v18 += (v56 + (1 << v26) - 1) & -(1 << v26);
            }
            goto LABEL_22;
          }
          v26 = -1;
          v27 = 1LL << v22;
        }
        v18 = -(int)v27 & (v27 + v61);
        goto LABEL_17;
      }
      v66 = sub_BDB740((__int64)v59, *(_QWORD *)(*v16 + 8));
      v67 = v35;
      v36 = sub_CA1930(&v66);
      if ( v57 )
      {
        _BitScanReverse64(&v37, v57);
        v38 = 0x8000000000000000LL >> ((unsigned __int8)v37 ^ 0x3Fu);
        v39 = -(__int64)v38;
        v18 = -(int)v38 & (v38 + v61);
        if ( *v59 && v36 < v57 )
          v18 = v54 - v36 + (-(int)v38 & (v38 + v61));
      }
      else
      {
        LODWORD(v39) = 0;
        LODWORD(v38) = 0;
        v18 = 0;
      }
      if ( v64 <= v17 )
        break;
LABEL_22:
      ++v17;
      v16 += 4;
      if ( v16 == v58 )
        goto LABEL_33;
    }
    v40 = v18 + v36;
    if ( v18 + (unsigned int)v36 <= 0x320 )
    {
      v62 = v18 + v36;
      v65 = v39;
      v41 = sub_2464620((__int64)a1, (unsigned int **)a3, v18);
      LODWORD(v39) = v65;
      v40 = v62;
      v42 = v41;
      if ( v41 )
      {
        LOBYTE(v43) = byte_4FE8EA8;
        HIBYTE(v43) = 1;
        v53 = v43;
        v44 = sub_246F3F0(a1[3], *v16);
        sub_2463EC0((__int64 *)a3, v44, v42, v53, 0);
        v40 = v62;
        LODWORD(v39) = v65;
      }
    }
    ++v17;
    v16 += 4;
    v18 = v39 & (v38 + v40 - 1);
  }
  while ( v16 != v58 );
LABEL_33:
  v45 = v18;
LABEL_34:
  v46 = sub_AD64C0(*(_QWORD *)(a1[2] + 80), v45, 0);
  return sub_2463EC0((__int64 *)a3, v46, *(_QWORD *)(a1[2] + 152), 0, 0);
}
