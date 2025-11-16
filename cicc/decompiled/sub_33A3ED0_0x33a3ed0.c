// Function: sub_33A3ED0
// Address: 0x33a3ed0
//
void __fastcall sub_33A3ED0(__int64 a1, __int64 a2)
{
  int v3; // edx
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r13
  __int64 v9; // r12
  unsigned int v10; // edx
  int v11; // edx
  __int64 (__fastcall *v12)(__int64, __int64, __int64, __int64); // rdx
  __int64 (__fastcall *v13)(__int64, __int64, unsigned int); // rax
  int v14; // edx
  int v15; // r9d
  unsigned __int16 v16; // ax
  __int128 v17; // rax
  int v18; // r9d
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // r12
  __int64 v22; // r10
  _BYTE *v23; // rsi
  _QWORD *v24; // rax
  __int64 v25; // rsi
  int v26; // r9d
  __int64 v27; // r10
  __int64 v28; // rdx
  __int64 v29; // rax
  unsigned __int16 v30; // dx
  __int64 v31; // r8
  bool v32; // al
  __int64 v33; // rcx
  __int64 v34; // rax
  __int64 v35; // r11
  unsigned __int64 v36; // rcx
  __int64 v37; // rsi
  __int64 v38; // rdx
  __int64 v39; // r9
  __int64 v40; // rax
  __int64 v41; // r8
  unsigned __int64 v42; // rdx
  __int64 v43; // rax
  __int16 v44; // si
  __int64 v45; // rax
  int v46; // esi
  int v47; // edx
  bool v48; // al
  unsigned __int16 v49; // ax
  int v50; // edx
  __int128 v51; // [rsp-30h] [rbp-100h]
  __int64 v52; // [rsp+10h] [rbp-C0h]
  __int64 v53; // [rsp+10h] [rbp-C0h]
  __int64 v54; // [rsp+18h] [rbp-B8h]
  __int64 v55; // [rsp+20h] [rbp-B0h]
  __int64 v56; // [rsp+20h] [rbp-B0h]
  __int64 v57; // [rsp+20h] [rbp-B0h]
  __int64 v58; // [rsp+30h] [rbp-A0h]
  __int128 v59; // [rsp+30h] [rbp-A0h]
  int v60; // [rsp+40h] [rbp-90h]
  __int64 v61; // [rsp+48h] [rbp-88h]
  int v62; // [rsp+48h] [rbp-88h]
  __int64 v63; // [rsp+50h] [rbp-80h]
  __int64 v64; // [rsp+50h] [rbp-80h]
  int v65; // [rsp+50h] [rbp-80h]
  unsigned int v66; // [rsp+58h] [rbp-78h]
  __int64 v67; // [rsp+80h] [rbp-50h] BYREF
  int v68; // [rsp+88h] [rbp-48h]
  __int64 v69; // [rsp+90h] [rbp-40h] BYREF
  __int64 v70; // [rsp+98h] [rbp-38h]

  v3 = *(_DWORD *)(a1 + 848);
  v4 = *(_QWORD *)a1;
  v67 = 0;
  v68 = v3;
  if ( v4 )
  {
    if ( &v67 != (__int64 *)(v4 + 48) )
    {
      v5 = *(_QWORD *)(v4 + 48);
      v67 = v5;
      if ( v5 )
        sub_B96E90((__int64)&v67, v5, 1);
    }
  }
  v61 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 864) + 40LL));
  v6 = sub_338B750(a1, *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v8 = v7;
  v9 = v6;
  v63 = sub_338B750(a1, *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
  v66 = v10;
  v55 = v61;
  v58 = *(_QWORD *)(*(_QWORD *)(a1 + 864) + 16LL);
  v60 = sub_2D5BAE0(v58, v61, *(__int64 **)(a2 + 8), 0);
  v62 = v11;
  v12 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v58 + 72LL);
  if ( v12 == sub_2FE4D20 )
  {
    v13 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v58 + 32LL);
    if ( v13 == sub_2D42F30 )
    {
      v14 = sub_AE2980(v55, 0)[1];
      v16 = 2;
      if ( v14 != 1 )
      {
        v16 = 3;
        if ( v14 != 2 )
        {
          v16 = 4;
          if ( v14 != 4 )
          {
            v16 = 5;
            if ( v14 != 8 )
            {
              v16 = 6;
              if ( v14 != 16 )
              {
                v16 = 7;
                if ( v14 != 32 )
                {
                  v16 = 8;
                  if ( v14 != 64 )
                    v16 = 9 * (v14 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v16 = v13(v58, v55, 0);
    }
  }
  else
  {
    v16 = ((__int64 (__fastcall *)(__int64, __int64))v12)(v58, v55);
  }
  *(_QWORD *)&v17 = sub_33FAF80(*(_QWORD *)(a1 + 864), 498, (unsigned int)&v67, v16, 0, v15);
  *((_QWORD *)&v51 + 1) = v8;
  *(_QWORD *)&v51 = v9;
  v19 = sub_3406EB0(*(_QWORD *)(a1 + 864), 158, (unsigned int)&v67, v60, v62, v18, v51, v17);
  v21 = v20;
  v22 = v19;
  v23 = *(_BYTE **)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( (unsigned __int8)(*v23 - 12) > 1u )
  {
    v56 = v19;
    *(_QWORD *)&v59 = sub_338B750(a1, (__int64)v23);
    v27 = v56;
    *((_QWORD *)&v59 + 1) = v28;
    v29 = *(_QWORD *)(v63 + 48) + 16LL * v66;
    v31 = *(_QWORD *)(v29 + 8);
    LOWORD(v69) = *(_WORD *)v29;
    v30 = v69;
    v70 = v31;
    if ( (_WORD)v69 )
    {
      if ( (unsigned __int16)(v69 - 17) <= 0xD3u )
      {
        LODWORD(v31) = 0;
        v30 = word_4456580[(unsigned __int16)v69 - 1];
      }
    }
    else
    {
      v52 = v31;
      v32 = sub_30070B0((__int64)&v69);
      v30 = 0;
      LODWORD(v31) = v52;
      v27 = v56;
      if ( v32 )
      {
        v49 = sub_3009970((__int64)&v69, v63, 0, v33, v52);
        v27 = v56;
        LODWORD(v31) = v50;
        v30 = v49;
      }
    }
    v57 = v27;
    v34 = sub_33FAF80(*(_QWORD *)(a1 + 864), 385, (unsigned int)&v67, v30, v31, v26);
    v35 = *(_QWORD *)(a1 + 864);
    v36 = v21;
    v37 = v34;
    v39 = v38;
    v40 = (unsigned int)v38;
    v41 = v37;
    v42 = v57;
    v43 = *(_QWORD *)(v37 + 48) + 16 * v40;
    v44 = *(_WORD *)v43;
    v45 = *(_QWORD *)(v43 + 8);
    LOWORD(v69) = v44;
    v70 = v45;
    if ( v44 )
    {
      v46 = ((unsigned __int16)(v44 - 17) < 0xD4u) + 205;
    }
    else
    {
      v53 = v41;
      v54 = v39;
      v65 = v35;
      v48 = sub_30070B0((__int64)&v69);
      LODWORD(v35) = v65;
      v41 = v53;
      v39 = v54;
      v42 = v57;
      v36 = v21;
      v46 = 205 - (!v48 - 1);
    }
    v22 = sub_340EC60(v35, v46, (unsigned int)&v67, v60, v62, 0, v41, v39, __PAIR128__(v36, v42), v59);
    LODWORD(v21) = v47;
  }
  v64 = v22;
  v69 = a2;
  v24 = sub_337DC20(a1 + 8, &v69);
  *v24 = v64;
  v25 = v67;
  *((_DWORD *)v24 + 2) = v21;
  if ( v25 )
    sub_B91220((__int64)&v67, v25);
}
