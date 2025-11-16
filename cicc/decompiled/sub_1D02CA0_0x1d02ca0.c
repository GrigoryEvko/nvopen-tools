// Function: sub_1D02CA0
// Address: 0x1d02ca0
//
__int64 __fastcall sub_1D02CA0(__int64 a1, __int64 a2, char a3, __int64 a4)
{
  int v4; // r15d
  int v8; // ecx
  int v9; // r8d
  int v10; // r9d
  int v12; // eax
  __int64 v13; // rdi
  __int64 (*v14)(); // rax
  bool v15; // cc
  __int64 v16; // rdi
  __int64 (*v17)(); // rax
  __int64 v18; // rdi
  __int64 (*v19)(); // rax
  int v20; // eax
  int v21; // r11d
  int v22; // r13d
  int v23; // eax
  unsigned __int16 v24; // ax
  int v25; // eax
  int v26; // eax
  int v27; // eax
  __int64 v28; // rdi
  __int64 (*v29)(); // rax
  int v30; // eax
  int v31; // eax
  int v32; // [rsp+8h] [rbp-48h]
  int v33; // [rsp+14h] [rbp-3Ch]
  int v34; // [rsp+14h] [rbp-3Ch]
  int v35; // [rsp+14h] [rbp-3Ch]
  int v36; // [rsp+14h] [rbp-3Ch]
  int v37; // [rsp+14h] [rbp-3Ch]
  int v38; // [rsp+14h] [rbp-3Ch]
  int v39; // [rsp+18h] [rbp-38h]
  int v40; // [rsp+18h] [rbp-38h]
  int v41; // [rsp+18h] [rbp-38h]
  int v42; // [rsp+18h] [rbp-38h]
  int v43; // [rsp+18h] [rbp-38h]
  int v44; // [rsp+18h] [rbp-38h]
  int v45; // [rsp+18h] [rbp-38h]
  int v46; // [rsp+1Ch] [rbp-34h]
  int v47; // [rsp+1Ch] [rbp-34h]
  int v48; // [rsp+1Ch] [rbp-34h]
  int v49; // [rsp+1Ch] [rbp-34h]
  int v50; // [rsp+1Ch] [rbp-34h]
  int v51; // [rsp+1Ch] [rbp-34h]
  int v52; // [rsp+1Ch] [rbp-34h]
  int v53; // [rsp+1Ch] [rbp-34h]
  int v54; // [rsp+1Ch] [rbp-34h]
  int v55; // [rsp+1Ch] [rbp-34h]

  v4 = 0;
  if ( (*(_BYTE *)(a1 + 228) & 1) == 0 )
    v4 = (unsigned __int8)sub_1D01030(a1);
  v8 = 0;
  if ( (*(_BYTE *)(a2 + 228) & 1) == 0 )
    v8 = (unsigned __int8)sub_1D01030(a2);
  if ( (*(_BYTE *)(a1 + 236) & 2) == 0 )
  {
    v47 = v8;
    sub_1F01F70(a1);
    v8 = v47;
  }
  v9 = v4 + *(_DWORD *)(a1 + 244);
  if ( (*(_BYTE *)(a2 + 236) & 2) == 0 )
  {
    v39 = v4 + *(_DWORD *)(a1 + 244);
    v46 = v8;
    sub_1F01F70(a2);
    v9 = v39;
    v8 = v46;
  }
  v10 = v8 + *(_DWORD *)(a2 + 244);
  if ( !a3 )
  {
    v12 = *(_DWORD *)(a4 + 8);
    if ( v9 <= v12 )
    {
      v13 = *(_QWORD *)(*(_QWORD *)(a4 + 88) + 704LL);
      v14 = *(__int64 (**)())(*(_QWORD *)v13 + 24LL);
      if ( v14 == sub_1D00B90 )
      {
LABEL_15:
        if ( v10 > *(_DWORD *)(a4 + 8) )
          return 0xFFFFFFFFLL;
        if ( v14 != sub_1D00B90 )
        {
          v34 = v8 + *(_DWORD *)(a2 + 244);
          v41 = v9;
          v51 = v8;
          v25 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v14)(v13, a2, 0);
          v8 = v51;
          v9 = v41;
          v10 = v34;
          if ( v25 )
            return 0xFFFFFFFFLL;
        }
        goto LABEL_17;
      }
      v33 = v8 + *(_DWORD *)(a2 + 244);
      v40 = v9;
      v48 = v8;
      v20 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v14)(v13, a1, 0);
      v8 = v48;
      v9 = v40;
      v10 = v33;
      v21 = v20;
      goto LABEL_37;
    }
LABEL_23:
    if ( v10 <= v12 )
    {
      v16 = *(_QWORD *)(*(_QWORD *)(a4 + 88) + 704LL);
      v17 = *(__int64 (**)())(*(_QWORD *)v16 + 24LL);
      if ( v17 == sub_1D00B90 )
        return 1;
      v35 = v8 + *(_DWORD *)(a2 + 244);
      v42 = v9;
      v52 = v8;
      v26 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v17)(v16, a2, 0);
      v8 = v52;
      v9 = v42;
      v10 = v35;
      if ( !v26 )
        return 1;
    }
    goto LABEL_39;
  }
  if ( *(_DWORD *)(a1 + 232) != 4 )
  {
    if ( *(_DWORD *)(a2 + 232) != 4 )
      return 0;
    if ( v10 > *(_DWORD *)(a4 + 8) )
      return 0xFFFFFFFFLL;
    v18 = *(_QWORD *)(*(_QWORD *)(a4 + 88) + 704LL);
    v19 = *(__int64 (**)())(*(_QWORD *)v18 + 24LL);
    if ( v19 != sub_1D00B90 )
    {
      v37 = v8 + *(_DWORD *)(a2 + 244);
      v44 = v9;
      v54 = v8;
      v30 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v19)(v18, a2, 0);
      v8 = v54;
      v9 = v44;
      v10 = v37;
      if ( v30 )
        return 0xFFFFFFFFLL;
    }
LABEL_33:
    if ( *(_DWORD *)(a1 + 232) != 4 && *(_DWORD *)(a2 + 232) != 4 )
      return 0;
LABEL_18:
    if ( !*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a4 + 88) + 704LL) + 8LL) )
    {
      v15 = v9 <= v10;
      if ( v9 != v10 )
      {
LABEL_20:
        if ( v15 )
          return 0xFFFFFFFFLL;
        return 1;
      }
    }
    goto LABEL_41;
  }
  v12 = *(_DWORD *)(a4 + 8);
  if ( v9 > v12 )
  {
    if ( *(_DWORD *)(a2 + 232) != 4 )
      return 1;
    goto LABEL_23;
  }
  v13 = *(_QWORD *)(*(_QWORD *)(a4 + 88) + 704LL);
  v14 = *(__int64 (**)())(*(_QWORD *)v13 + 24LL);
  if ( v14 == sub_1D00B90 )
  {
    if ( *(_DWORD *)(a2 + 232) != 4 )
    {
LABEL_29:
      if ( *(_DWORD *)(a1 + 232) != 4 )
        return 0;
      goto LABEL_18;
    }
    goto LABEL_15;
  }
  v36 = v8 + *(_DWORD *)(a2 + 244);
  v43 = v9;
  v53 = v8;
  v27 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v14)(v13, a1, 0);
  v8 = v53;
  v9 = v43;
  v10 = v36;
  v21 = v27;
  if ( *(_DWORD *)(a2 + 232) != 4 )
  {
    if ( v27 )
      return 1;
    goto LABEL_29;
  }
LABEL_37:
  if ( v10 <= *(_DWORD *)(a4 + 8) )
  {
    v28 = *(_QWORD *)(*(_QWORD *)(a4 + 88) + 704LL);
    v29 = *(__int64 (**)())(*(_QWORD *)v28 + 24LL);
    if ( v29 == sub_1D00B90 )
      goto LABEL_62;
    v32 = v21;
    v38 = v10;
    v45 = v9;
    v55 = v8;
    v31 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v29)(v28, a2, 0);
    v8 = v55;
    v9 = v45;
    v10 = v38;
    v21 = v32;
    if ( !v31 )
    {
LABEL_62:
      if ( v21 )
        return 1;
LABEL_17:
      if ( !a3 )
        goto LABEL_18;
      goto LABEL_33;
    }
  }
  if ( !v21 )
    return 0xFFFFFFFFLL;
LABEL_39:
  v15 = v9 <= v10;
  if ( v9 != v10 )
    goto LABEL_20;
  if ( !a3 || *(_DWORD *)(a1 + 232) == 4 || *(_DWORD *)(a2 + 232) == 4 )
  {
LABEL_41:
    if ( (*(_BYTE *)(a1 + 236) & 1) == 0 )
    {
      v50 = v8;
      sub_1F01DD0(a1);
      v8 = v50;
    }
    v22 = *(_DWORD *)(a1 + 240) - v4;
    if ( (*(_BYTE *)(a2 + 236) & 1) == 0 )
    {
      v49 = v8;
      sub_1F01DD0(a2);
      v8 = v49;
    }
    v23 = *(_DWORD *)(a2 + 240) - v8;
    if ( v22 == v23 )
    {
      v24 = *(_WORD *)(a2 + 226);
      if ( *(_WORD *)(a1 + 226) == v24 )
        return 0;
      if ( *(_WORD *)(a1 + 226) <= v24 )
        return 0xFFFFFFFFLL;
    }
    else if ( v22 >= v23 )
    {
      return 0xFFFFFFFFLL;
    }
    return 1;
  }
  return 0;
}
