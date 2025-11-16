// Function: sub_2C33CD0
// Address: 0x2c33cd0
//
void __fastcall sub_2C33CD0(__int64 *a1)
{
  __int64 v1; // rdx
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rbx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // r13
  char v29; // al
  char *v30; // r14
  __int64 v31; // r15
  __int64 (__fastcall *v32)(__int64); // rax
  __int64 v33; // rbx
  __int64 v34; // rsi
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // r12
  __int64 v38; // r13
  __int64 v39; // rdx
  __int64 v40; // rsi
  __int64 v41; // r15
  __int64 v42; // r8
  __int64 v43; // r9
  __int64 v44; // rsi
  __int64 v45; // rsi
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rsi
  unsigned __int64 v50; // rsi
  unsigned __int64 v51; // rdx
  char *v52; // r14
  unsigned int v53; // eax
  unsigned int v54; // edx
  __int64 v55; // rcx
  __int64 v56; // rcx
  char v57; // di
  char v58; // cl
  _BYTE *v59; // [rsp+0h] [rbp-500h]
  _BYTE *v60; // [rsp+8h] [rbp-4F8h]
  _QWORD *v61; // [rsp+10h] [rbp-4F0h]
  _BYTE *v62; // [rsp+18h] [rbp-4E8h]
  _QWORD *v63; // [rsp+20h] [rbp-4E0h]
  _BYTE *v64; // [rsp+28h] [rbp-4D8h]
  _QWORD *v65; // [rsp+30h] [rbp-4D0h]
  __int64 v66; // [rsp+48h] [rbp-4B8h]
  __int64 v67; // [rsp+50h] [rbp-4B0h]
  __int64 *v68; // [rsp+58h] [rbp-4A8h]
  __int64 v69; // [rsp+60h] [rbp-4A0h]
  __int64 v70; // [rsp+68h] [rbp-498h]
  __int64 v71; // [rsp+70h] [rbp-490h] BYREF
  __int64 v72; // [rsp+78h] [rbp-488h] BYREF
  __int64 v73; // [rsp+80h] [rbp-480h] BYREF
  __int64 v74; // [rsp+88h] [rbp-478h] BYREF
  _QWORD v75[12]; // [rsp+90h] [rbp-470h] BYREF
  __int64 v76; // [rsp+F0h] [rbp-410h]
  __int64 v77; // [rsp+F8h] [rbp-408h]
  __int16 v78; // [rsp+108h] [rbp-3F8h]
  _QWORD v79[12]; // [rsp+110h] [rbp-3F0h] BYREF
  __int64 v80; // [rsp+170h] [rbp-390h]
  __int64 v81; // [rsp+178h] [rbp-388h]
  __int16 v82; // [rsp+188h] [rbp-378h]
  __int16 v83; // [rsp+198h] [rbp-368h]
  _QWORD v84[12]; // [rsp+1A0h] [rbp-360h] BYREF
  __int64 v85; // [rsp+200h] [rbp-300h]
  __int64 v86; // [rsp+208h] [rbp-2F8h]
  __int16 v87; // [rsp+218h] [rbp-2E8h] BYREF
  _QWORD v88[15]; // [rsp+220h] [rbp-2E0h] BYREF
  __int16 v89; // [rsp+298h] [rbp-268h]
  __int16 v90; // [rsp+2A8h] [rbp-258h]
  _BYTE v91[120]; // [rsp+2B0h] [rbp-250h] BYREF
  __int16 v92; // [rsp+328h] [rbp-1D8h]
  _BYTE v93[120]; // [rsp+330h] [rbp-1D0h] BYREF
  __int16 v94; // [rsp+3A8h] [rbp-158h]
  __int16 v95; // [rsp+3B8h] [rbp-148h]
  _BYTE v96[120]; // [rsp+3C0h] [rbp-140h] BYREF
  __int16 v97; // [rsp+438h] [rbp-C8h]
  _BYTE v98[120]; // [rsp+440h] [rbp-C0h] BYREF
  __int16 v99; // [rsp+4B8h] [rbp-48h]
  __int16 v100; // [rsp+4C8h] [rbp-38h]

  v63 = v84;
  sub_2C2F4B0(v84, *a1);
  v59 = v91;
  sub_2C31060((__int64)v91, (__int64)v84, v1, v2, v3, v4);
  sub_2AB1B50((__int64)&v87);
  sub_2AB1B50((__int64)v84);
  sub_2ABCC20(v75, (__int64)v91, v5, v6, v7, v8);
  v60 = v93;
  v78 = v92;
  v61 = v79;
  sub_2ABCC20(v79, (__int64)v93, v9, v10, v11, v12);
  v82 = v94;
  v83 = v95;
  v62 = v96;
  sub_2ABCC20(v84, (__int64)v96, v13, v14, v15, v16);
  v64 = v98;
  v87 = v97;
  v65 = v88;
  sub_2ABCC20(v88, (__int64)v98, v17, v18, v19, v20);
  v21 = v76;
  v89 = v99;
  v90 = v100;
  v22 = v77;
LABEL_2:
  v23 = v85;
  if ( v22 - v21 != v86 - v85 )
  {
LABEL_3:
    v24 = *(_QWORD *)(v22 - 32);
    v66 = sub_2BF05A0(v24);
    v70 = *(_QWORD *)(v24 + 120);
    if ( v70 == v66 )
    {
      while ( 1 )
      {
LABEL_39:
        sub_2AD7320((__int64)v75, v23, v21, v25, v26, v27);
        v22 = v77;
        v21 = v76;
        v23 = v80;
        if ( v77 - v76 == v81 - v80 )
        {
          if ( v76 == v77 )
            goto LABEL_2;
          v56 = v76;
          while ( *(_QWORD *)v56 == *(_QWORD *)v23 )
          {
            v57 = *(_BYTE *)(v56 + 24);
            if ( v57 != *(_BYTE *)(v23 + 24)
              || v57 && (*(_QWORD *)(v56 + 8) != *(_QWORD *)(v23 + 8) || *(_QWORD *)(v56 + 16) != *(_QWORD *)(v23 + 16)) )
            {
              break;
            }
            v56 += 32;
            v23 += 32;
            if ( v77 == v56 )
              goto LABEL_2;
          }
        }
        v25 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v77 - 32) + 8LL) - 1;
        if ( (unsigned int)v25 <= 1 )
          goto LABEL_2;
      }
    }
    while ( 1 )
    {
      v28 = v70;
      v70 = *(_QWORD *)(v70 + 8);
      v29 = *(_BYTE *)(v28 - 16);
      if ( v29 == 29 )
        break;
      if ( v29 == 31 )
      {
        v69 = 12;
        v30 = "evl.based.iv";
LABEL_9:
        v31 = 0;
        if ( *(_DWORD *)(v28 + 32) )
          v31 = **(_QWORD **)(v28 + 24);
        v68 = (__int64 *)(v28 - 24);
        v32 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)(v28 - 24) + 40LL);
        if ( v32 == sub_2AA7530 )
          v33 = *(_QWORD *)(*(_QWORD *)(v28 + 24) + 8LL);
        else
          v33 = ((__int64 (__fastcall *)(__int64 *, __int64, __int64, __int64, __int64, __int64, _BYTE *, _BYTE *, _QWORD *, _BYTE *, _QWORD *, _BYTE *, _QWORD *))v32)(
                  v68,
                  v23,
                  v21,
                  v25,
                  v26,
                  v27,
                  v59,
                  v60,
                  v61,
                  v62,
                  v63,
                  v64,
                  v65);
        v34 = *(_QWORD *)(v28 + 64);
        v71 = v34;
        if ( v34 )
          sub_B96E90((__int64)&v71, v34, 1);
        v37 = sub_22077B0(0xB8u);
        v67 = v28 + 72;
        if ( !v37 )
        {
          v41 = v71;
          if ( v71 )
          {
            v41 = 0;
            sub_B91220((__int64)&v71, v71);
          }
          sub_2C19D60(0, (__int64)v68);
          goto LABEL_38;
        }
        v72 = v71;
        if ( v71 )
        {
          sub_B96E90((__int64)&v72, v71, 1);
          v73 = v72;
          if ( v72 )
          {
            sub_B96E90((__int64)&v73, v72, 1);
            v74 = v73;
            if ( v73 )
              sub_B96E90((__int64)&v74, v73, 1);
            goto LABEL_20;
          }
        }
        else
        {
          v73 = 0;
        }
        v74 = 0;
LABEL_20:
        *(_BYTE *)(v37 + 8) = 35;
        v38 = v37 + 40;
        *(_QWORD *)(v37 + 24) = 0;
        *(_QWORD *)(v37 + 64) = v31;
        *(_QWORD *)v37 = &unk_4A231A8;
        *(_QWORD *)(v37 + 32) = 0;
        *(_QWORD *)(v37 + 16) = 0;
        *(_QWORD *)(v37 + 40) = &unk_4A23170;
        *(_QWORD *)(v37 + 48) = v37 + 64;
        *(_QWORD *)(v37 + 56) = 0x200000001LL;
        v39 = *(unsigned int *)(v31 + 24);
        if ( v39 + 1 > (unsigned __int64)*(unsigned int *)(v31 + 28) )
        {
          sub_C8D5F0(v31 + 16, (const void *)(v31 + 32), v39 + 1, 8u, v35, v36);
          v39 = *(unsigned int *)(v31 + 24);
        }
        *(_QWORD *)(*(_QWORD *)(v31 + 16) + 8 * v39) = v38;
        v40 = v74;
        ++*(_DWORD *)(v31 + 24);
        *(_QWORD *)(v37 + 80) = 0;
        *(_QWORD *)v37 = &unk_4A23A70;
        *(_QWORD *)(v37 + 40) = &unk_4A23AA8;
        *(_QWORD *)(v37 + 88) = v40;
        if ( v40 )
        {
          sub_B96E90(v37 + 88, v40, 1);
          if ( v74 )
            sub_B91220((__int64)&v74, v74);
        }
        v41 = v37 + 96;
        sub_2BF0340(v37 + 96, 1, 0, v37, v35, v36);
        v44 = v73;
        *(_QWORD *)v37 = &unk_4A231C8;
        *(_QWORD *)(v37 + 40) = &unk_4A23200;
        *(_QWORD *)(v37 + 96) = &unk_4A23238;
        if ( v44 )
          sub_B91220((__int64)&v73, v44);
        v45 = v72;
        *(_QWORD *)v37 = &unk_4A23FE8;
        *(_QWORD *)(v37 + 40) = &unk_4A24030;
        *(_QWORD *)(v37 + 96) = &unk_4A24068;
        if ( v45 )
          sub_B91220((__int64)&v72, v45);
        *(_QWORD *)v37 = &unk_4A24E58;
        *(_QWORD *)(v37 + 40) = &unk_4A24EA8;
        v46 = v37 + 168;
        *(_QWORD *)(v37 + 96) = &unk_4A24EE0;
        *(_QWORD *)(v37 + 152) = v37 + 168;
        if ( (unsigned int)v69 >= 8 )
        {
          v50 = (v37 + 176) & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v37 + 168) = *(_QWORD *)v30;
          *(_QWORD *)(v46 + v69 - 8) = *(_QWORD *)&v30[v69 - 8];
          v51 = v46 - v50;
          v52 = &v30[-v51];
          if ( (((_DWORD)v51 + (_DWORD)v69) & 0xFFFFFFF8) >= 8 )
          {
            v53 = (v51 + v69) & 0xFFFFFFF8;
            v54 = 0;
            do
            {
              v55 = v54;
              v54 += 8;
              *(_QWORD *)(v50 + v55) = *(_QWORD *)&v52[v55];
            }
            while ( v54 < v53 );
          }
        }
        else
        {
          *(_DWORD *)(v37 + 168) = *(_DWORD *)v30;
          *(_DWORD *)(v46 + (unsigned int)v69 - 4) = *(_DWORD *)&v30[(unsigned int)v69 - 4];
        }
        *(_QWORD *)(v37 + 160) = v69;
        *(_BYTE *)(v37 + v69 + 168) = 0;
        v47 = *(unsigned int *)(v37 + 56);
        if ( v47 + 1 > (unsigned __int64)*(unsigned int *)(v37 + 60) )
        {
          sub_C8D5F0(v37 + 48, (const void *)(v37 + 64), v47 + 1, 8u, v42, v43);
          v47 = *(unsigned int *)(v37 + 56);
        }
        *(_QWORD *)(*(_QWORD *)(v37 + 48) + 8 * v47) = v33;
        ++*(_DWORD *)(v37 + 56);
        v48 = *(unsigned int *)(v33 + 24);
        if ( v48 + 1 > (unsigned __int64)*(unsigned int *)(v33 + 28) )
        {
          sub_C8D5F0(v33 + 16, (const void *)(v33 + 32), v48 + 1, 8u, v42, v43);
          v48 = *(unsigned int *)(v33 + 24);
        }
        *(_QWORD *)(*(_QWORD *)(v33 + 16) + 8 * v48) = v38;
        v49 = v71;
        ++*(_DWORD *)(v33 + 24);
        if ( v49 )
          sub_B91220((__int64)&v71, v49);
        sub_2C19D60((_QWORD *)v37, (__int64)v68);
LABEL_38:
        v23 = v41;
        sub_2BF1250(v67, v41);
        sub_2C19E60(v68);
        if ( v70 == v66 )
          goto LABEL_39;
      }
      else if ( v70 == v66 )
      {
        goto LABEL_39;
      }
    }
    v69 = 5;
    v30 = "index";
    goto LABEL_9;
  }
  if ( v22 != v21 )
  {
    while ( *(_QWORD *)v21 == *(_QWORD *)v23 )
    {
      v58 = *(_BYTE *)(v21 + 24);
      if ( v58 != *(_BYTE *)(v23 + 24)
        || v58 && (*(_QWORD *)(v21 + 8) != *(_QWORD *)(v23 + 8) || *(_QWORD *)(v21 + 16) != *(_QWORD *)(v23 + 16)) )
      {
        break;
      }
      v21 += 32;
      v23 += 32;
      if ( v21 == v22 )
        goto LABEL_64;
    }
    goto LABEL_3;
  }
LABEL_64:
  sub_2AB1B50((__int64)v65);
  sub_2AB1B50((__int64)v63);
  sub_2AB1B50((__int64)v61);
  sub_2AB1B50((__int64)v75);
  sub_2AB1B50((__int64)v64);
  sub_2AB1B50((__int64)v62);
  sub_2AB1B50((__int64)v60);
  sub_2AB1B50((__int64)v59);
}
