// Function: sub_33532E0
// Address: 0x33532e0
//
__int64 __fastcall sub_33532E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r15d
  char v7; // r14
  __int64 v10; // rbx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  unsigned __int8 v14; // al
  __int64 v15; // rdx
  __int64 v17; // rdi
  __int64 (*v18)(); // rax
  __int64 v19; // rdi
  __int64 (*v20)(); // rax
  bool v21; // cc
  __int64 v22; // rdi
  __int64 (*v23)(); // rax
  __int64 v24; // rdi
  __int64 (*v25)(); // r11
  int v26; // eax
  int v27; // eax
  int v28; // r11d
  unsigned __int8 v29; // al
  int v30; // r13d
  int v31; // eax
  unsigned __int16 v32; // ax
  int v33; // eax
  __int64 v34; // rdi
  __int64 (*v35)(); // rax
  int v36; // eax
  int v37; // eax
  unsigned int v38; // [rsp+8h] [rbp-48h]
  unsigned int v39; // [rsp+10h] [rbp-40h]
  unsigned int v40; // [rsp+14h] [rbp-3Ch]
  unsigned int v41; // [rsp+14h] [rbp-3Ch]
  unsigned int v42; // [rsp+14h] [rbp-3Ch]
  unsigned int v43; // [rsp+14h] [rbp-3Ch]
  unsigned int v44; // [rsp+14h] [rbp-3Ch]
  unsigned int v45; // [rsp+18h] [rbp-38h]
  unsigned int v46; // [rsp+18h] [rbp-38h]
  unsigned int v47; // [rsp+18h] [rbp-38h]
  bool v48; // [rsp+18h] [rbp-38h]
  unsigned int v49; // [rsp+18h] [rbp-38h]
  unsigned int v50; // [rsp+18h] [rbp-38h]
  unsigned int v51; // [rsp+1Ch] [rbp-34h]
  unsigned int v52; // [rsp+1Ch] [rbp-34h]
  unsigned int v53; // [rsp+1Ch] [rbp-34h]
  int v54; // [rsp+1Ch] [rbp-34h]
  int v55; // [rsp+1Ch] [rbp-34h]
  unsigned int v56; // [rsp+1Ch] [rbp-34h]
  unsigned int v57; // [rsp+1Ch] [rbp-34h]
  unsigned int v58; // [rsp+1Ch] [rbp-34h]
  unsigned int v59; // [rsp+1Ch] [rbp-34h]

  v6 = 0;
  v7 = a3;
  v10 = a2;
  if ( (*(_BYTE *)(a1 + 248) & 1) == 0 )
    v6 = (unsigned __int8)sub_33516E0(a1);
  v11 = 0;
  if ( (*(_BYTE *)(a2 + 248) & 1) == 0 )
    v11 = (unsigned __int8)sub_33516E0(a2);
  if ( (*(_BYTE *)(a1 + 254) & 2) == 0 )
  {
    v52 = v11;
    sub_2F8F770(a1, (_QWORD *)a2, a3, v11, a5, a6);
    v11 = v52;
  }
  v12 = (unsigned int)(v6 + *(_DWORD *)(a1 + 244));
  if ( (*(_BYTE *)(a2 + 254) & 2) == 0 )
  {
    v45 = v6 + *(_DWORD *)(a1 + 244);
    v51 = v11;
    sub_2F8F770(a2, (_QWORD *)a2, a3, v11, v12, a6);
    v12 = v45;
    v11 = v51;
  }
  v13 = (unsigned int)(v11 + *(_DWORD *)(a2 + 244));
  if ( !v7 )
  {
    v15 = *(unsigned int *)(a4 + 8);
    if ( (int)v12 <= (int)v15 )
    {
      v17 = *(_QWORD *)(*(_QWORD *)(a4 + 88) + 672LL);
      v18 = *(__int64 (**)())(*(_QWORD *)v17 + 24LL);
      if ( v18 == sub_2EC0B50 )
        goto LABEL_15;
      a2 = a1;
      v42 = v13;
      v47 = v12;
      v57 = v11;
      v33 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v18)(v17, a1, 0);
      v11 = v57;
      v12 = v47;
      v13 = v42;
      v28 = v33;
      goto LABEL_50;
    }
LABEL_23:
    if ( (int)v13 <= (int)v15 )
    {
      v22 = *(_QWORD *)(*(_QWORD *)(a4 + 88) + 672LL);
      v23 = *(__int64 (**)())(*(_QWORD *)v22 + 24LL);
      if ( v23 == sub_2EC0B50 )
        return 1;
      v44 = v11 + *(_DWORD *)(a2 + 244);
      v50 = v12;
      v59 = v11;
      v37 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v23)(v22, a2, 0);
      v11 = v59;
      v12 = v50;
      v13 = v44;
      if ( !v37 )
        return 1;
    }
    goto LABEL_36;
  }
  v14 = *(_BYTE *)(a1 + 254);
  v15 = v14 & 0xF0;
  if ( (v14 & 0xF0) != 0x40 )
  {
    if ( (*(_BYTE *)(a2 + 254) & 0xF0) != 0x40 )
      return 0;
LABEL_15:
    if ( (int)v13 > *(_DWORD *)(a4 + 8) )
      return 0xFFFFFFFFLL;
    v19 = *(_QWORD *)(*(_QWORD *)(a4 + 88) + 672LL);
    v20 = *(__int64 (**)())(*(_QWORD *)v19 + 24LL);
    if ( v20 != sub_2EC0B50 )
    {
      v40 = v11 + *(_DWORD *)(a2 + 244);
      v46 = v12;
      v53 = v11;
      v26 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v20)(v19, a2, 0);
      v11 = v53;
      v12 = v46;
      v13 = v40;
      if ( v26 )
        return 0xFFFFFFFFLL;
    }
LABEL_17:
    if ( !v7 || (*(_BYTE *)(a1 + 254) & 0xF0) == 0x40 )
      goto LABEL_18;
    v14 = *(_BYTE *)(v10 + 254);
LABEL_29:
    if ( (v14 & 0xF0) != 0x40 )
      return 0;
LABEL_18:
    if ( !*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a4 + 88) + 672LL) + 8LL) )
    {
      v21 = (int)v12 <= (int)v13;
      if ( (_DWORD)v12 != (_DWORD)v13 )
      {
LABEL_20:
        if ( v21 )
          return 0xFFFFFFFFLL;
        return 1;
      }
    }
LABEL_38:
    v29 = *(_BYTE *)(a1 + 254);
    goto LABEL_39;
  }
  v15 = *(unsigned int *)(a4 + 8);
  if ( (int)v12 > (int)v15 )
  {
    if ( (*(_BYTE *)(a2 + 254) & 0xF0) != 0x40 )
      return 1;
    goto LABEL_23;
  }
  v24 = *(_QWORD *)(*(_QWORD *)(a4 + 88) + 672LL);
  v25 = *(__int64 (**)())(*(_QWORD *)v24 + 24LL);
  if ( v25 == sub_2EC0B50 )
  {
    v15 = *(_BYTE *)(a2 + 254) & 0xF0;
    if ( (*(_BYTE *)(a2 + 254) & 0xF0) != 0x40 )
      goto LABEL_29;
    goto LABEL_15;
  }
  a2 = a1;
  v43 = v13;
  v49 = v12;
  v58 = v11;
  v36 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v25)(v24, a1, 0);
  v11 = v58;
  v12 = v49;
  v28 = v36;
  v13 = v43;
  if ( (*(_BYTE *)(v10 + 254) & 0xF0) != 0x40 )
  {
    if ( v36 )
      return 1;
    v14 = *(_BYTE *)(a1 + 254);
    goto LABEL_29;
  }
LABEL_50:
  if ( (int)v13 <= *(_DWORD *)(a4 + 8) )
  {
    v48 = v28 != 0;
    v34 = *(_QWORD *)(*(_QWORD *)(a4 + 88) + 672LL);
    v35 = *(__int64 (**)())(*(_QWORD *)v34 + 24LL);
    if ( v35 == sub_2EC0B50 )
      goto LABEL_52;
    v39 = v13;
    a2 = v10;
    v38 = v12;
    v41 = v11;
    v54 = v28;
    v27 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v35)(v34, v10, 0);
    v11 = v41;
    v12 = v38;
    v13 = v39;
    if ( !v27 )
    {
LABEL_52:
      if ( v48 )
        return 1;
      goto LABEL_17;
    }
    v28 = v54;
  }
  if ( !v28 )
    return 0xFFFFFFFFLL;
LABEL_36:
  v21 = (int)v12 <= (int)v13;
  if ( (_DWORD)v12 != (_DWORD)v13 )
    goto LABEL_20;
  if ( !v7 )
    goto LABEL_38;
  v29 = *(_BYTE *)(a1 + 254);
  v15 = v29 & 0xF0;
  if ( (v29 & 0xF0) == 0x40 || (v15 = *(_BYTE *)(v10 + 254) & 0xF0, (*(_BYTE *)(v10 + 254) & 0xF0) == 0x40) )
  {
LABEL_39:
    if ( (v29 & 1) == 0 )
    {
      v56 = v11;
      sub_2F8F5D0(a1, (_QWORD *)a2, v15, v11, v12, v13);
      v11 = v56;
    }
    v30 = *(_DWORD *)(a1 + 240) - v6;
    if ( (*(_BYTE *)(v10 + 254) & 1) == 0 )
    {
      v55 = v11;
      sub_2F8F5D0(v10, (_QWORD *)a2, v15, v11, v12, v13);
      LODWORD(v11) = v55;
    }
    v31 = *(_DWORD *)(v10 + 240) - v11;
    if ( v30 == v31 )
    {
      v32 = *(_WORD *)(v10 + 252);
      if ( *(_WORD *)(a1 + 252) == v32 )
        return 0;
      if ( *(_WORD *)(a1 + 252) <= v32 )
        return 0xFFFFFFFFLL;
    }
    else if ( v30 >= v31 )
    {
      return 0xFFFFFFFFLL;
    }
    return 1;
  }
  return 0;
}
