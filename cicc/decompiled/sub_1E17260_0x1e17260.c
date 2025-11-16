// Function: sub_1E17260
// Address: 0x1e17260
//
bool __fastcall sub_1E17260(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // rdi
  __int64 v13; // r14
  __int64 (*v14)(); // rax
  __int64 v15; // rax
  __int64 v16; // r14
  __int16 v17; // dx
  __int64 (*v18)(); // rax
  bool v20; // al
  __int64 v21; // rax
  __int16 v22; // dx
  bool v23; // al
  __int64 *v24; // rcx
  _QWORD *v25; // r8
  __int64 v26; // r15
  __int64 v27; // rdi
  __int64 v28; // rdx
  __int64 v29; // r11
  __int64 v30; // r9
  __int64 v31; // r13
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // r10
  bool v34; // al
  unsigned __int64 v35; // r13
  unsigned __int64 v36; // r13
  char v37; // al
  __int64 v38; // r15
  __int64 v39; // rdx
  __int64 v40; // rdi
  __int64 v41; // rsi
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rcx
  __int64 v46; // r13
  unsigned __int64 v47; // rdi
  char v48; // al
  unsigned __int64 v49; // [rsp+8h] [rbp-D8h]
  __int64 v50; // [rsp+10h] [rbp-D0h]
  __int64 v51; // [rsp+18h] [rbp-C8h]
  _QWORD *v52; // [rsp+20h] [rbp-C0h]
  __int64 v53; // [rsp+20h] [rbp-C0h]
  __int64 *v54; // [rsp+28h] [rbp-B8h]
  __int64 v55; // [rsp+28h] [rbp-B8h]
  __int64 v56; // [rsp+30h] [rbp-B0h]
  _QWORD *v57; // [rsp+30h] [rbp-B0h]
  unsigned __int64 v58; // [rsp+38h] [rbp-A8h]
  __int64 *v59; // [rsp+38h] [rbp-A8h]
  char v60; // [rsp+40h] [rbp-A0h]
  __int64 v61; // [rsp+40h] [rbp-A0h]
  __int64 v62; // [rsp+48h] [rbp-98h]
  __int64 v63; // [rsp+48h] [rbp-98h]
  __int64 v64; // [rsp+48h] [rbp-98h]
  _QWORD v65[6]; // [rsp+50h] [rbp-90h] BYREF
  unsigned __int64 v66; // [rsp+80h] [rbp-60h] BYREF
  __int64 v67; // [rsp+88h] [rbp-58h]
  __int64 v68; // [rsp+90h] [rbp-50h]
  __int64 v69; // [rsp+98h] [rbp-48h]
  __int64 v70; // [rsp+A0h] [rbp-40h]

  v8 = sub_1E15F70(a1);
  v11 = 0;
  v12 = *(_QWORD *)(v8 + 16);
  v13 = v8;
  v14 = *(__int64 (**)())(*(_QWORD *)v12 + 40LL);
  if ( v14 != sub_1D00B00 )
    v11 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, _QWORD))v14)(v12, a2, v9, v10, 0);
  v15 = *(_QWORD *)(a1 + 16);
  v16 = *(_QWORD *)(v13 + 56);
  if ( *(_WORD *)v15 != 1 || (*(_BYTE *)(*(_QWORD *)(a1 + 32) + 64LL) & 0x10) == 0 )
  {
    v17 = *(_WORD *)(a1 + 46);
    if ( (v17 & 4) != 0 || (v17 & 8) == 0 )
    {
      if ( (*(_QWORD *)(v15 + 8) & 0x20000LL) != 0 )
        goto LABEL_7;
    }
    else
    {
      v62 = v11;
      v20 = sub_1E15D00(a1, 0x20000u, 1);
      v11 = v62;
      if ( v20 )
        goto LABEL_7;
    }
    v21 = *(_QWORD *)(a3 + 16);
    if ( *(_WORD *)v21 != 1 || (*(_BYTE *)(*(_QWORD *)(a3 + 32) + 64LL) & 0x10) == 0 )
    {
      v22 = *(_WORD *)(a3 + 46);
      if ( (v22 & 4) != 0 || (v22 & 8) == 0 )
      {
        v23 = (*(_QWORD *)(v21 + 8) & 0x20000LL) != 0;
      }
      else
      {
        v64 = v11;
        v23 = sub_1E15D00(a3, 0x20000u, 1);
        v11 = v64;
      }
      if ( !v23 )
        return 0;
    }
  }
LABEL_7:
  v18 = *(__int64 (**)())(*(_QWORD *)v11 + 952LL);
  if ( v18 != sub_1E15BA0 && ((unsigned __int8 (__fastcall *)(__int64, __int64, __int64, __int64))v18)(v11, a1, a3, a2) )
    return 0;
  if ( *(_BYTE *)(a1 + 49) != 1 || *(_BYTE *)(a3 + 49) != 1 )
    return 1;
  v24 = **(__int64 ***)(a1 + 56);
  v25 = **(_QWORD ***)(a3 + 56);
  v26 = v24[1];
  v27 = *v24;
  v28 = v25[1];
  v29 = v26;
  v30 = v25[3];
  v63 = v24[3];
  if ( v28 <= v26 )
    v29 = v25[1];
  v60 = (v27 >> 2) & 1;
  if ( ((v27 >> 2) & 1) != 0 )
  {
    v46 = *v25;
    if ( (*v25 & 4) == 0 )
    {
      v35 = v46 & 0xFFFFFFFFFFFFFFF8LL;
      v33 = 0;
      v47 = v27 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v47 )
      {
        if ( v35 )
        {
          v53 = v29;
          v55 = v25[1];
          v57 = v25;
          v59 = v24;
          v61 = v25[3];
          v48 = (*(__int64 (__fastcall **)(unsigned __int64, __int64))(*(_QWORD *)v47 + 40LL))(v47, v16);
          v30 = v61;
          v24 = v59;
          v25 = v57;
          v28 = v55;
          v29 = v53;
          v33 = 0;
          if ( !v48 )
            return 0;
        }
      }
      goto LABEL_38;
    }
    v34 = 0;
    v33 = 0;
    v36 = v46 & 0xFFFFFFFFFFFFFFF8LL;
    v58 = v27 & 0xFFFFFFFFFFFFFFF8LL;
    v60 = (v27 & 0xFFFFFFFFFFFFFFF8LL) != 0;
    goto LABEL_32;
  }
  v31 = *v25;
  v32 = v27 & 0xFFFFFFFFFFFFFFF8LL;
  v33 = v32;
  v34 = v32 != 0;
  if ( (*v25 & 4) != 0 )
  {
    v58 = 0;
    v36 = v31 & 0xFFFFFFFFFFFFFFF8LL;
LABEL_32:
    if ( v36 && v34 )
    {
      v49 = v33;
      v50 = v29;
      v51 = v25[1];
      v52 = v25;
      v54 = v24;
      v56 = v25[3];
      if ( !(*(unsigned __int8 (__fastcall **)(unsigned __int64, __int64))(*(_QWORD *)v36 + 40LL))(v36, v16) )
        return 0;
      v33 = v49;
      v29 = v50;
      v28 = v51;
      v25 = v52;
      v24 = v54;
      v30 = v56;
      v37 = v60 & (v58 == v36);
    }
    else
    {
      v37 = v60 & (v36 != 0 && v58 == v36);
    }
    if ( v37 )
      goto LABEL_25;
    v35 = 0;
    goto LABEL_38;
  }
  v35 = v31 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v35 == v32 && v34 && v35 != 0 )
  {
LABEL_25:
    if ( v28 < v26 )
      v28 = v26;
    else
      v30 = v63;
    return v29 + v30 > v28;
  }
LABEL_38:
  if ( !a2 || !v33 || !v35 )
    return 1;
  v38 = v63 + v26 - v29;
  v39 = v30 + v28 - v29;
  if ( a4 )
  {
    v40 = v25[5];
    v41 = v25[6];
    v66 = v35;
    v42 = v25[7];
    v67 = v39;
    v68 = v40;
    v43 = v24[6];
    v69 = v41;
    v70 = v42;
    v44 = v24[5];
    v45 = v24[7];
  }
  else
  {
    v67 = v39;
    v45 = 0;
    v43 = 0;
    v44 = 0;
    v66 = v35;
    v68 = 0;
    v69 = 0;
    v70 = 0;
  }
  v65[3] = v43;
  v65[0] = v33;
  v65[1] = v38;
  v65[2] = v44;
  v65[4] = v45;
  return (unsigned __int8)sub_134CB50(a2, (__int64)v65, (__int64)&v66) != 0;
}
