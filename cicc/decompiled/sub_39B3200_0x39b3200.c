// Function: sub_39B3200
// Address: 0x39b3200
//
__int64 __fastcall sub_39B3200(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v9; // r15
  __int64 v10; // r13
  __int64 v11; // rdx
  __int64 v12; // r9
  __int64 v13; // r11
  _QWORD *v14; // rax
  __int64 (__fastcall *v15)(__int64, __int64 *, __int64, __int64, __int64, __int64); // rax
  __int64 v16; // rsi
  unsigned int v17; // r12d
  __int64 (__fastcall *v19)(__int64, __int64, __int64); // rax
  __int64 v20; // rax
  __int64 (__fastcall *v21)(_QWORD *, __int64, __int64, __int64); // rax
  __int64 v22; // rax
  __int64 (__fastcall *v23)(__int64, _QWORD, __int64, __int64, __int64); // rax
  __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // r14
  __int64 (__fastcall *v27)(__int64, __int64, __int64); // rax
  __int64 v28; // rax
  __int64 (__fastcall *v29)(_QWORD *, __int64, __int64, __int64); // rax
  __int64 v30; // rax
  __int64 v31; // r15
  unsigned __int8 v32; // cl
  char v33; // al
  _BOOL4 v34; // r13d
  __int64 v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rbx
  __int64 (__fastcall *v38)(__int64); // rax
  __int64 *v39; // r9
  __int64 v40; // rax
  __int64 v41; // rax
  unsigned __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // rax
  void (__fastcall *v45)(__int64, __int64, __int64, _BOOL4); // rax
  __int64 v46; // rdi
  __int64 v47; // r13
  __int64 (__fastcall *v48)(__int64); // rax
  __int64 *v49; // r14
  __int64 v50; // rax
  __int64 v51; // rax
  unsigned __int64 v52; // r15
  __int64 v53; // rdx
  __int64 v54; // rax
  __int64 v55; // rbx
  void (__fastcall *v56)(__int64); // rax
  __int64 v57; // r11
  char v58; // al
  _DWORD *v59; // rax
  _DWORD *v60; // rsi
  __int64 v61; // rax
  __int64 v62; // r11
  __int64 v63; // rdi
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // [rsp-10h] [rbp-100h]
  unsigned __int8 v67; // [rsp+14h] [rbp-DCh]
  unsigned __int8 v68; // [rsp+18h] [rbp-D8h]
  unsigned __int64 v69; // [rsp+18h] [rbp-D8h]
  __int64 v70; // [rsp+18h] [rbp-D8h]
  __int64 v71; // [rsp+20h] [rbp-D0h]
  __int64 v72; // [rsp+20h] [rbp-D0h]
  __int64 v73; // [rsp+20h] [rbp-D0h]
  _DWORD *v74; // [rsp+20h] [rbp-D0h]
  __int64 v75; // [rsp+20h] [rbp-D0h]
  __int64 v76; // [rsp+20h] [rbp-D0h]
  __int64 v77; // [rsp+20h] [rbp-D0h]
  __int64 v78; // [rsp+28h] [rbp-C8h]
  __int64 v79; // [rsp+28h] [rbp-C8h]
  __int64 v80; // [rsp+28h] [rbp-C8h]
  __int64 v81; // [rsp+28h] [rbp-C8h]
  __int64 v82; // [rsp+28h] [rbp-C8h]
  __int64 *v83; // [rsp+28h] [rbp-C8h]
  __int64 v84; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v85; // [rsp+28h] [rbp-C8h]
  _DWORD *v86; // [rsp+28h] [rbp-C8h]
  __int64 *v87; // [rsp+28h] [rbp-C8h]
  __int64 v88; // [rsp+28h] [rbp-C8h]
  __int64 v90; // [rsp+30h] [rbp-C0h]
  __int64 v92; // [rsp+48h] [rbp-A8h] BYREF
  __int64 v93; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v94; // [rsp+58h] [rbp-98h] BYREF
  __int64 v95[2]; // [rsp+60h] [rbp-90h] BYREF
  __int16 v96; // [rsp+70h] [rbp-80h]
  unsigned __int64 v97[2]; // [rsp+80h] [rbp-70h] BYREF
  __int64 v98; // [rsp+90h] [rbp-60h] BYREF

  if ( (*(_BYTE *)(a1 + 840) & 0x40) != 0 )
    *(_BYTE *)(a6 + 1162) = 0;
  v9 = *(_QWORD **)(a1 + 8);
  v10 = *(_QWORD *)(a1 + 632);
  v92 = 0;
  v11 = *(_QWORD *)(a1 + 608);
  v12 = *(_QWORD *)(a1 + 616);
  v13 = *(_QWORD *)(a1 + 624);
  v14 = v9;
  switch ( (_DWORD)a5 )
  {
    case 1:
      v19 = (__int64 (__fastcall *)(__int64, __int64, __int64))v9[17];
      if ( v19 )
      {
        v71 = *(_QWORD *)(a1 + 616);
        v20 = v19(v13, v12, a6);
        v9 = *(_QWORD **)(a1 + 8);
        v12 = v71;
        v78 = v20;
      }
      else
      {
        v78 = 0;
      }
      v21 = (__int64 (__fastcall *)(_QWORD *, __int64, __int64, __int64))v9[12];
      if ( !v21 )
        goto LABEL_16;
      v22 = v21(v9, v10, v12, a1 + 840);
      if ( !v78 )
        goto LABEL_16;
      v74 = (_DWORD *)v22;
      if ( !v22 )
        goto LABEL_16;
      *(_BYTE *)(a6 + 1163) = 0;
      v96 = 260;
      v95[0] = a1 + 472;
      sub_16E1010((__int64)v97, (__int64)v95);
      v57 = *(_QWORD *)(a1 + 8);
      v58 = *(_BYTE *)(a1 + 841);
      v95[0] = v78;
      v68 = v58 & 1;
      v67 = (*(_BYTE *)(a1 + 840) & 2) != 0;
      v59 = v74;
      v60 = v74;
      v75 = v57;
      v86 = v59;
      if ( a4 )
      {
        sub_390A1E0((__int64)&v94, v60, a3, a4);
        v61 = (__int64)v86;
        v62 = v75;
      }
      else
      {
        sub_390A0A0((__int64)&v94, v60, a3);
        v62 = v75;
        v61 = (__int64)v86;
      }
      v93 = v61;
      v63 = v92;
      v92 = sub_39B29F0(v62, (__int64)v97, a6, (__int64)&v93, (__int64)&v94, (__int64)v95, v10, v67, v68, 1u);
      if ( v63 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v63 + 48LL))(v63);
      if ( v93 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v93 + 8LL))(v93);
      if ( v94 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v94 + 8LL))(v94);
      if ( v95[0] )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v95[0] + 8LL))(v95[0]);
      if ( (__int64 *)v97[0] != &v98 )
      {
        j_j___libc_free_0(v97[0]);
        v14 = *(_QWORD **)(a1 + 8);
        break;
      }
      goto LABEL_63;
    case 2:
      v55 = sub_39F42D0(a6, a2, v11, a4, a5, v12);
      v56 = (void (__fastcall *)(__int64))v9[22];
      if ( v56 )
        v56(v55);
      v92 = v55;
      goto LABEL_63;
    case 0:
      v23 = (__int64 (__fastcall *)(__int64, _QWORD, __int64, __int64, __int64))v9[16];
      v24 = a1 + 472;
      if ( v23 )
      {
        v72 = *(_QWORD *)(a1 + 616);
        v79 = v13;
        v25 = v23(v24, *(unsigned int *)(v11 + 168), v11, v13, v12);
        v9 = *(_QWORD **)(a1 + 8);
        v13 = v79;
        v12 = v72;
        v26 = v25;
      }
      else
      {
        v26 = 0;
      }
      v93 = 0;
      if ( (*(_BYTE *)(a1 + 841) & 4) != 0 )
      {
        v27 = (__int64 (__fastcall *)(__int64, __int64, __int64))v9[17];
        if ( v27 )
        {
          v80 = v12;
          v28 = v27(v13, v12, a6);
          v12 = v80;
          v93 = v28;
          v9 = *(_QWORD **)(a1 + 8);
        }
      }
      v29 = (__int64 (__fastcall *)(_QWORD *, __int64, __int64, __int64))v9[12];
      if ( v29 )
        v30 = v29(v9, v10, v12, a1 + 840);
      else
        v30 = 0;
      v94 = v30;
      sub_39B30C0(v95, a3);
      v31 = v95[0];
      v32 = *(_BYTE *)(a1 + 840);
      v90 = *(_QWORD *)(a1 + 8);
      v33 = *(_BYTE *)(a1 + 841);
      v97[0] = v95[0];
      v34 = (v33 & 0x10) != 0;
      v95[0] = 0;
      v35 = sub_39E81C0(a6, (unsigned int)v97, v34, v32 >> 7, v26, (unsigned int)&v93, (__int64)&v94, (v33 & 8) != 0);
      a5 = v97[0];
      v36 = v66;
      v37 = v35;
      if ( v97[0] )
      {
        v38 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v97[0] + 8LL);
        if ( v38 == sub_16BE0F0 )
        {
          *(_QWORD *)v97[0] = &unk_49EF340;
          if ( *(_QWORD *)(a5 + 24) != *(_QWORD *)(a5 + 8) )
          {
            v81 = a5;
            sub_16E7BA0((__int64 *)a5);
            a5 = v81;
          }
          v39 = *(__int64 **)(a5 + 40);
          if ( v39 )
          {
            v40 = *(_QWORD *)(a5 + 8);
            if ( !*(_DWORD *)(a5 + 32) || v40 )
            {
              v42 = *(_QWORD *)(a5 + 16) - v40;
            }
            else
            {
              v82 = a5;
              v41 = sub_16E7720();
              a5 = v82;
              v42 = v41;
              v39 = *(__int64 **)(v82 + 40);
            }
            v43 = v39[3];
            v44 = v39[1];
            if ( v42 )
            {
              if ( v43 != v44 )
              {
                v69 = v42;
                v76 = a5;
                v87 = v39;
                sub_16E7BA0(v39);
                v42 = v69;
                a5 = v76;
                v39 = v87;
              }
              v77 = a5;
              v70 = (__int64)v39;
              v88 = v42;
              v64 = sub_2207820(v42);
              sub_16E7A40(v70, v64, v88, 1);
              a5 = v77;
            }
            else
            {
              if ( v43 != v44 )
              {
                v73 = a5;
                v83 = v39;
                sub_16E7BA0(v39);
                a5 = v73;
                v39 = v83;
              }
              v84 = a5;
              sub_16E7A40((__int64)v39, 0, 0, 0);
              a5 = v84;
            }
          }
          v85 = a5;
          sub_16E7960(a5);
          v36 = 64;
          j_j___libc_free_0(v85);
        }
        else
        {
          ((void (__fastcall *)(unsigned __int64, __int64))v38)(v97[0], v66);
        }
      }
      v45 = *(void (__fastcall **)(__int64, __int64, __int64, _BOOL4))(v90 + 184);
      if ( v45 )
      {
        v36 = v31;
        v45(v37, v31, v26, v34);
      }
      v46 = v92;
      v92 = v37;
      if ( v46 )
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v46 + 48LL))(v46, v36);
      v47 = v95[0];
      if ( v95[0] )
      {
        v48 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v95[0] + 8LL);
        if ( v48 == sub_16BE0F0 )
        {
          *(_QWORD *)v95[0] = &unk_49EF340;
          if ( *(_QWORD *)(v47 + 24) != *(_QWORD *)(v47 + 8) )
            sub_16E7BA0((__int64 *)v47);
          v49 = *(__int64 **)(v47 + 40);
          if ( v49 )
          {
            v50 = *(_QWORD *)(v47 + 8);
            if ( !*(_DWORD *)(v47 + 32) || v50 )
            {
              v52 = *(_QWORD *)(v47 + 16) - v50;
            }
            else
            {
              v51 = sub_16E7720();
              v49 = *(__int64 **)(v47 + 40);
              v52 = v51;
            }
            v53 = v49[3];
            v54 = v49[1];
            if ( v52 )
            {
              if ( v53 != v54 )
                sub_16E7BA0(v49);
              v65 = sub_2207820(v52);
              sub_16E7A40((__int64)v49, v65, v52, 1);
            }
            else
            {
              if ( v53 != v54 )
                sub_16E7BA0(v49);
              sub_16E7A40((__int64)v49, 0, 0, 0);
            }
          }
          sub_16E7960(v47);
          v36 = 64;
          j_j___libc_free_0(v47);
        }
        else
        {
          ((void (__fastcall *)(__int64, __int64))v48)(v95[0], v36);
        }
      }
      if ( v94 )
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v94 + 8LL))(v94, v36);
      if ( v93 )
      {
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v93 + 8LL))(v93, v36);
        v14 = *(_QWORD **)(a1 + 8);
        break;
      }
LABEL_63:
      v14 = *(_QWORD **)(a1 + 8);
      break;
  }
  v15 = (__int64 (__fastcall *)(__int64, __int64 *, __int64, __int64, __int64, __int64))v14[14];
  if ( v15 )
  {
    v16 = v15(a1, &v92, v11, a4, a5, v12);
    if ( v16 )
    {
      v17 = 0;
      (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a2 + 16LL))(a2, v16, 1);
      goto LABEL_9;
    }
  }
LABEL_16:
  v17 = 1;
LABEL_9:
  if ( v92 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v92 + 48LL))(v92);
  return v17;
}
