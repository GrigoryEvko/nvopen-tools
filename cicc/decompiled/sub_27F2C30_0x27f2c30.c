// Function: sub_27F2C30
// Address: 0x27f2c30
//
void __fastcall sub_27F2C30(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned int v3; // edx
  __int64 v4; // r14
  __int64 v5; // r13
  const char *v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rdx
  int v9; // edx
  int v10; // ebx
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // r15
  _QWORD *v14; // r11
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 *v18; // rbx
  __int64 v19; // r14
  __int64 *v20; // r12
  int v21; // eax
  unsigned int v22; // edi
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rdi
  __int64 v26; // r15
  int v27; // eax
  char v28; // al
  __int64 v29; // r11
  const char *v30; // rax
  __int64 v31; // rdi
  __int64 v32; // rdx
  int v33; // edx
  int v34; // r15d
  __int64 v35; // rax
  __int64 v36; // rsi
  __int64 v37; // r11
  __int64 v38; // rbx
  _QWORD *v39; // r10
  __int64 v40; // rax
  __int64 *v41; // r14
  __int64 v42; // rdx
  __int64 *v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r15
  __int64 v46; // r13
  __int64 *v47; // r12
  int v48; // eax
  unsigned int v49; // edi
  __int64 v50; // rax
  __int64 v51; // rdi
  __int64 v52; // rdi
  __int64 v53; // rbx
  int v54; // eax
  __int64 v55; // rax
  __int64 v56; // r15
  __int64 v57; // rbx
  _QWORD *v58; // rax
  __int64 v59; // r14
  __int16 v60; // ax
  const char **v61; // r15
  const char *v62; // rsi
  _QWORD *v63; // rdi
  __int64 v64; // rcx
  __int64 *v65; // rsi
  __int64 v66; // rsi
  unsigned __int8 *v67; // rsi
  __int64 *v68; // [rsp-D0h] [rbp-D0h]
  __int64 v69; // [rsp-C8h] [rbp-C8h]
  __int64 v70; // [rsp-C8h] [rbp-C8h]
  __int64 v71; // [rsp-C0h] [rbp-C0h]
  __int64 v72; // [rsp-C0h] [rbp-C0h]
  __int64 v73; // [rsp-C0h] [rbp-C0h]
  __int64 v74; // [rsp-A8h] [rbp-A8h]
  __int64 v75; // [rsp-A0h] [rbp-A0h]
  __int64 v76; // [rsp-98h] [rbp-98h]
  __int64 v77; // [rsp-90h] [rbp-90h]
  __int64 v78; // [rsp-88h] [rbp-88h]
  _QWORD *v79; // [rsp-88h] [rbp-88h]
  __int64 v80; // [rsp-88h] [rbp-88h]
  __int64 v81; // [rsp-88h] [rbp-88h]
  _QWORD *v82; // [rsp-88h] [rbp-88h]
  __int64 v83; // [rsp-88h] [rbp-88h]
  __int64 v84; // [rsp-88h] [rbp-88h]
  __int64 v85; // [rsp-88h] [rbp-88h]
  __int64 v86; // [rsp-70h] [rbp-70h]
  char *v87; // [rsp-68h] [rbp-68h] BYREF
  __int64 v88; // [rsp-60h] [rbp-60h]
  const char *v89; // [rsp-58h] [rbp-58h]
  __int16 v90; // [rsp-48h] [rbp-48h]

  if ( *(_BYTE *)(a1 + 128) )
  {
    v1 = a1;
    v2 = *(_QWORD *)(a1 + 24);
    v3 = *(_DWORD *)(v2 + 8);
    if ( v3 )
    {
      v86 = 0;
      v68 = (__int64 *)(a1 + 88);
      v76 = v3;
      v77 = 0;
      while ( 1 )
      {
        v4 = *(_QWORD *)(*(_QWORD *)v2 + 8 * v86);
        v5 = sub_11D7E40(*(__int64 **)(v1 + 8), v4);
        if ( (unsigned __int8)sub_D495C0(*(_QWORD *)(v1 + 64), v5, v4) )
        {
          v6 = sub_BD5D20(v5);
          v7 = *(_QWORD *)(v1 + 48);
          v90 = 773;
          v87 = (char *)v6;
          v88 = v8;
          v89 = ".lcssa";
          sub_102DBD0(v7, v4);
          v10 = v9;
          v78 = *(_QWORD *)(v5 + 8);
          v11 = sub_BD2DA0(80);
          v12 = v78;
          v13 = v11;
          if ( v11 )
          {
            v79 = (_QWORD *)v11;
            sub_B44260(v11, v12, 55, 0x8000000u, 0, 0);
            *(_DWORD *)(v13 + 72) = v10;
            sub_BD6B50((unsigned __int8 *)v13, (const char **)&v87);
            sub_BD2A10(v13, *(_DWORD *)(v13 + 72), 1);
            v14 = v79;
          }
          else
          {
            v14 = 0;
          }
          v15 = v74;
          LOWORD(v15) = 1;
          v74 = v15;
          sub_B44220(v14, *(_QWORD *)(v4 + 56), v15);
          v16 = sub_102DBD0(*(_QWORD *)(v1 + 48), v4);
          v18 = (__int64 *)v16;
          if ( v16 != v16 + 8 * v17 )
          {
            v71 = v4;
            v19 = v13;
            v69 = v1;
            v20 = (__int64 *)(v16 + 8 * v17);
            do
            {
              v26 = *v18;
              v27 = *(_DWORD *)(v19 + 4) & 0x7FFFFFF;
              if ( v27 == *(_DWORD *)(v19 + 72) )
              {
                sub_B48D90(v19);
                v27 = *(_DWORD *)(v19 + 4) & 0x7FFFFFF;
              }
              v21 = (v27 + 1) & 0x7FFFFFF;
              v22 = v21 | *(_DWORD *)(v19 + 4) & 0xF8000000;
              v23 = *(_QWORD *)(v19 - 8) + 32LL * (unsigned int)(v21 - 1);
              *(_DWORD *)(v19 + 4) = v22;
              if ( *(_QWORD *)v23 )
              {
                v24 = *(_QWORD *)(v23 + 8);
                **(_QWORD **)(v23 + 16) = v24;
                if ( v24 )
                  *(_QWORD *)(v24 + 16) = *(_QWORD *)(v23 + 16);
              }
              *(_QWORD *)v23 = v5;
              v25 = *(_QWORD *)(v5 + 16);
              *(_QWORD *)(v23 + 8) = v25;
              if ( v25 )
                *(_QWORD *)(v25 + 16) = v23 + 8;
              *(_QWORD *)(v23 + 16) = v5 + 16;
              ++v18;
              *(_QWORD *)(v5 + 16) = v23;
              *(_QWORD *)(*(_QWORD *)(v19 - 8)
                        + 32LL * *(unsigned int *)(v19 + 72)
                        + 8LL * ((*(_DWORD *)(v19 + 4) & 0x7FFFFFFu) - 1)) = v26;
            }
            while ( v20 != v18 );
            v13 = v19;
            v1 = v69;
            v4 = v71;
          }
          v5 = v13;
        }
        v80 = *(_QWORD *)(v1 + 16);
        v28 = sub_D495C0(*(_QWORD *)(v1 + 64), v80, v4);
        v29 = v80;
        if ( v28 )
        {
          v30 = sub_BD5D20(v80);
          v31 = *(_QWORD *)(v1 + 48);
          v87 = (char *)v30;
          v90 = 773;
          v88 = v32;
          v89 = ".lcssa";
          sub_102DBD0(v31, v4);
          v34 = v33;
          v72 = v80;
          v81 = *(_QWORD *)(v80 + 8);
          v35 = sub_BD2DA0(80);
          v36 = v81;
          v37 = v72;
          v38 = v35;
          if ( v35 )
          {
            v82 = (_QWORD *)v35;
            sub_B44260(v35, v36, 55, 0x8000000u, 0, 0);
            *(_DWORD *)(v38 + 72) = v34;
            sub_BD6B50((unsigned __int8 *)v38, (const char **)&v87);
            sub_BD2A10(v38, *(_DWORD *)(v38 + 72), 1);
            v37 = v72;
            v39 = v82;
          }
          else
          {
            v39 = 0;
          }
          v40 = v75;
          v83 = v37;
          LOWORD(v40) = 1;
          v75 = v40;
          sub_B44220(v39, *(_QWORD *)(v4 + 56), v40);
          v41 = (__int64 *)sub_102DBD0(*(_QWORD *)(v1 + 48), v4);
          v43 = &v41[v42];
          v44 = v83 + 16;
          if ( v41 != v43 )
          {
            v73 = v5;
            v45 = v83;
            v46 = v38;
            v70 = v1;
            v47 = v43;
            do
            {
              v53 = *v41;
              v54 = *(_DWORD *)(v46 + 4) & 0x7FFFFFF;
              if ( v54 == *(_DWORD *)(v46 + 72) )
              {
                v84 = v44;
                sub_B48D90(v46);
                v44 = v84;
                v54 = *(_DWORD *)(v46 + 4) & 0x7FFFFFF;
              }
              v48 = (v54 + 1) & 0x7FFFFFF;
              v49 = v48 | *(_DWORD *)(v46 + 4) & 0xF8000000;
              v50 = *(_QWORD *)(v46 - 8) + 32LL * (unsigned int)(v48 - 1);
              *(_DWORD *)(v46 + 4) = v49;
              if ( *(_QWORD *)v50 )
              {
                v51 = *(_QWORD *)(v50 + 8);
                **(_QWORD **)(v50 + 16) = v51;
                if ( v51 )
                  *(_QWORD *)(v51 + 16) = *(_QWORD *)(v50 + 16);
              }
              *(_QWORD *)v50 = v45;
              v52 = *(_QWORD *)(v45 + 16);
              *(_QWORD *)(v50 + 8) = v52;
              if ( v52 )
                *(_QWORD *)(v52 + 16) = v50 + 8;
              *(_QWORD *)(v50 + 16) = v44;
              ++v41;
              *(_QWORD *)(v45 + 16) = v50;
              *(_QWORD *)(*(_QWORD *)(v46 - 8)
                        + 32LL * *(unsigned int *)(v46 + 72)
                        + 8LL * ((*(_DWORD *)(v46 + 4) & 0x7FFFFFFu) - 1)) = v53;
            }
            while ( v47 != v41 );
            v38 = v46;
            v1 = v70;
            v5 = v73;
          }
          v29 = v38;
        }
        v85 = v29;
        v55 = **(_QWORD **)(v1 + 32) + 16 * v86;
        v56 = *(_QWORD *)v55;
        v57 = *(unsigned __int16 *)(v55 + 8);
        v58 = sub_BD2C40(80, unk_3F10A10);
        v59 = (__int64)v58;
        if ( v58 )
          sub_B4D460((__int64)v58, v5, v85, v56, v57);
        v60 = *(_WORD *)(v59 + 2);
        if ( *(_BYTE *)(v1 + 81) )
        {
          v60 &= 0xFC7Fu;
          LOBYTE(v60) = v60 | 0x80;
          *(_WORD *)(v59 + 2) = v60;
        }
        v61 = (const char **)(v59 + 48);
        *(_WORD *)(v59 + 2) = (2 * *(unsigned __int8 *)(v1 + 80)) | v60 & 0xFF81;
        v62 = *(const char **)(v1 + 72);
        v87 = (char *)v62;
        if ( !v62 )
          break;
        sub_B96E90((__int64)&v87, (__int64)v62, 1);
        if ( v61 == (const char **)&v87 )
        {
          if ( v87 )
            sub_B91220((__int64)&v87, (__int64)v87);
LABEL_42:
          if ( v86 )
            goto LABEL_43;
          goto LABEL_55;
        }
        v66 = *(_QWORD *)(v59 + 48);
        if ( v66 )
          goto LABEL_52;
LABEL_53:
        v67 = (unsigned __int8 *)v87;
        *(_QWORD *)(v59 + 48) = v87;
        if ( !v67 )
          goto LABEL_42;
        sub_B976B0((__int64)&v87, v67, v59 + 48);
        if ( v86 )
        {
LABEL_43:
          sub_B99FD0(v59, 0x26u, v77);
LABEL_44:
          if ( !*(_QWORD *)(v1 + 88) )
            goto LABEL_57;
          goto LABEL_45;
        }
LABEL_55:
        sub_AE9860(v59, *(_QWORD *)(v1 + 136), *(_QWORD *)(v1 + 144));
        v77 = 0;
        if ( (*(_BYTE *)(v59 + 7) & 0x20) == 0 )
          goto LABEL_44;
        v77 = sub_B91C10(v59, 38);
        if ( !*(_QWORD *)(v1 + 88) )
        {
LABEL_57:
          if ( !*(_QWORD *)(v1 + 96) && !*(_QWORD *)(v1 + 104) && !*(_QWORD *)(v1 + 112) )
            goto LABEL_46;
        }
LABEL_45:
        sub_B9A100(v59, v68);
LABEL_46:
        v63 = *(_QWORD **)(v1 + 56);
        v64 = *(_QWORD *)(**(_QWORD **)(v1 + 40) + 8 * v86);
        if ( v64 )
          v65 = (__int64 *)sub_D69570(v63, v59, 0, v64);
        else
          v65 = (__int64 *)sub_D694D0(v63, v59, 0, *(_QWORD *)(v59 + 40), 0, 1u);
        *(_QWORD *)(**(_QWORD **)(v1 + 40) + 8 * v86) = v65;
        sub_D75120(*(__int64 **)(v1 + 56), v65, 1);
        if ( v76 == ++v86 )
          return;
        v2 = *(_QWORD *)(v1 + 24);
      }
      if ( v61 == (const char **)&v87 )
        goto LABEL_42;
      v66 = *(_QWORD *)(v59 + 48);
      if ( !v66 )
        goto LABEL_42;
LABEL_52:
      sub_B91220(v59 + 48, v66);
      goto LABEL_53;
    }
  }
}
