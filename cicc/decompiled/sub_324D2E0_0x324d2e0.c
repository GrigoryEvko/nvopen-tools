// Function: sub_324D2E0
// Address: 0x324d2e0
//
__int64 __fastcall sub_324D2E0(__int64 *a1, __int64 a2, unsigned __int64 a3, char a4)
{
  __int64 v7; // rbx
  unsigned __int8 v8; // al
  __int64 v9; // r15
  unsigned __int8 v10; // cl
  __int64 v11; // rcx
  unsigned __int8 v12; // si
  __int64 v13; // rcx
  __int64 v14; // rax
  unsigned __int8 v15; // dl
  __int64 v16; // rax
  __int64 v17; // rdx
  int v18; // esi
  unsigned __int8 v19; // si
  __int64 v20; // rsi
  unsigned __int8 v21; // al
  __int64 *v22; // rcx
  int v23; // eax
  __int64 v24; // rsi
  unsigned __int8 v25; // al
  __int64 *v26; // rcx
  unsigned int v27; // eax
  __int64 v28; // r8
  __int64 v29; // rdx
  __int64 v30; // rdx
  unsigned __int8 v31; // al
  _BYTE *v32; // rdi
  size_t v33; // rdx
  size_t v34; // rcx
  __int64 v35; // rax
  int v36; // esi
  __int64 v37; // r8
  int v38; // esi
  unsigned int v39; // edx
  __int64 *v40; // rax
  __int64 v41; // r11
  __int64 result; // rax
  unsigned __int8 v43; // al
  __int64 v44; // rdx
  __int64 v45; // rdi
  __int64 v46; // rdx
  __int64 *v47; // rdx
  __int64 v48; // rdx
  _QWORD *v49; // rcx
  int v50; // eax
  int v51; // ebx
  __int64 v52; // [rsp+8h] [rbp-58h]
  int v53; // [rsp+8h] [rbp-58h]
  unsigned __int8 *v54; // [rsp+10h] [rbp-50h]
  __int64 v55; // [rsp+18h] [rbp-48h]
  int v56; // [rsp+2Ch] [rbp-34h]

  v7 = a2 - 16;
  v8 = *(_BYTE *)(a2 - 16);
  if ( (v8 & 2) != 0 )
  {
    v9 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 48LL);
    if ( !v9 || a4 )
    {
      v55 = 0;
      v54 = 0;
LABEL_32:
      v29 = 0;
      if ( *(_DWORD *)(a2 - 24) <= 9u )
        goto LABEL_35;
      v30 = *(_QWORD *)(a2 - 32);
      goto LABEL_34;
    }
  }
  else
  {
    v9 = *(_QWORD *)(v7 - 8LL * ((v8 >> 2) & 0xF) + 48);
    if ( !v9 || a4 )
    {
      v55 = 0;
      v54 = 0;
      goto LABEL_52;
    }
  }
  v52 = v9 - 16;
  v10 = *(_BYTE *)(v9 - 16);
  if ( (v10 & 2) != 0 )
  {
    v11 = *(_QWORD *)(*(_QWORD *)(v9 - 32) + 32LL);
    v12 = *(_BYTE *)(v11 - 16);
    if ( (v12 & 2) != 0 )
    {
LABEL_6:
      v13 = *(_QWORD *)(*(_QWORD *)(v11 - 32) + 24LL);
      if ( (v8 & 2) != 0 )
        goto LABEL_7;
LABEL_56:
      v14 = *(_QWORD *)(v7 - 8LL * ((v8 >> 2) & 0xF) + 32);
      v15 = *(_BYTE *)(v14 - 16);
      if ( (v15 & 2) != 0 )
        goto LABEL_8;
LABEL_57:
      v16 = v14 - 16 - 8LL * ((v15 >> 2) & 0xF);
      goto LABEL_9;
    }
  }
  else
  {
    v11 = *(_QWORD *)(v52 - 8LL * ((v10 >> 2) & 0xF) + 32);
    v12 = *(_BYTE *)(v11 - 16);
    if ( (v12 & 2) != 0 )
      goto LABEL_6;
  }
  v13 = *(_QWORD *)(v11 - 16 - 8LL * ((v12 >> 2) & 0xF) + 24);
  if ( (v8 & 2) == 0 )
    goto LABEL_56;
LABEL_7:
  v14 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 32LL);
  v15 = *(_BYTE *)(v14 - 16);
  if ( (v15 & 2) == 0 )
    goto LABEL_57;
LABEL_8:
  v16 = *(_QWORD *)(v14 - 32);
LABEL_9:
  v17 = *(_QWORD *)(v16 + 24);
  if ( v13 )
  {
    v18 = (*(_BYTE *)(v13 - 16) & 2) != 0 ? *(_DWORD *)(v13 - 24) : (*(_WORD *)(v13 - 16) >> 6) & 0xF;
    if ( v17 && v18 )
    {
      v19 = *(_BYTE *)(v17 - 16);
      if ( (v19 & 2) != 0 )
      {
        if ( !*(_DWORD *)(v17 - 24) )
          goto LABEL_16;
        v47 = *(__int64 **)(v17 - 32);
      }
      else
      {
        if ( (*(_WORD *)(v17 - 16) & 0x3C0) == 0 )
          goto LABEL_16;
        v47 = (__int64 *)(v17 - 16 - 8LL * ((v19 >> 2) & 0xF));
      }
      v48 = *v47;
      if ( v48 )
      {
        v49 = (*(_BYTE *)(v13 - 16) & 2) != 0
            ? *(_QWORD **)(v13 - 32)
            : (_QWORD *)(v13 - 16 - 8LL * ((*(_BYTE *)(v13 - 16) >> 2) & 0xF));
        if ( v48 != *v49 || !*v49 )
          sub_32495E0(a1, a3, v48, 73);
      }
    }
  }
LABEL_16:
  v55 = 0;
  v54 = sub_3247C80((__int64)a1, (unsigned __int8 *)v9);
  if ( *(_BYTE *)(a1[26] + 3686) )
  {
    v43 = *(_BYTE *)(v9 - 16);
    if ( (v43 & 2) != 0 )
      v44 = *(_QWORD *)(v9 - 32);
    else
      v44 = v52 - 8LL * ((v43 >> 2) & 0xF);
    v45 = *(_QWORD *)(v44 + 24);
    if ( v45 )
    {
      sub_B91420(v45);
      v55 = v46;
    }
    else
    {
      v55 = 0;
    }
  }
  v20 = v9;
  if ( *(_BYTE *)v9 != 16 )
  {
    v21 = *(_BYTE *)(v9 - 16);
    if ( (v21 & 2) != 0 )
      v22 = *(__int64 **)(v9 - 32);
    else
      v22 = (__int64 *)(v52 - 8LL * ((v21 >> 2) & 0xF));
    v20 = *v22;
  }
  v23 = (*(__int64 (__fastcall **)(__int64 *, __int64))(*a1 + 80))(a1, v20);
  v24 = a2;
  v53 = v23;
  if ( *(_BYTE *)a2 != 16 )
  {
    v25 = *(_BYTE *)(a2 - 16);
    if ( (v25 & 2) != 0 )
      v26 = *(__int64 **)(a2 - 32);
    else
      v26 = (__int64 *)(v7 - 8LL * ((v25 >> 2) & 0xF));
    v24 = *v26;
  }
  v27 = (*(__int64 (__fastcall **)(__int64 *, __int64))(*a1 + 80))(a1, v24);
  if ( v53 != v27 )
  {
    BYTE2(v56) = 0;
    sub_3249A20(a1, (unsigned __int64 **)(a3 + 8), 58, v56, v27);
  }
  v28 = *(unsigned int *)(a2 + 16);
  if ( (_DWORD)v28 != *(_DWORD *)(v9 + 16) )
  {
    BYTE2(v56) = 0;
    sub_3249A20(a1, (unsigned __int64 **)(a3 + 8), 59, v56, v28);
  }
  v8 = *(_BYTE *)(a2 - 16);
  if ( (v8 & 2) != 0 )
    goto LABEL_32;
LABEL_52:
  if ( ((*(_WORD *)(a2 - 16) >> 6) & 0xFu) > 9 )
  {
    v30 = v7 - 8LL * ((v8 >> 2) & 0xF);
LABEL_34:
    v29 = *(_QWORD *)(v30 + 72);
    goto LABEL_35;
  }
  v29 = 0;
LABEL_35:
  sub_324D230(a1, a3, v29);
  v31 = *(_BYTE *)(a2 - 16);
  if ( (v31 & 2) != 0 )
  {
    v32 = *(_BYTE **)(*(_QWORD *)(a2 - 32) + 24LL);
    if ( v32 )
    {
LABEL_37:
      v32 = (_BYTE *)sub_B91420((__int64)v32);
      v34 = v33;
      goto LABEL_38;
    }
  }
  else
  {
    v32 = *(_BYTE **)(v7 - 8LL * ((v31 >> 2) & 0xF) + 24);
    if ( v32 )
      goto LABEL_37;
  }
  v34 = 0;
LABEL_38:
  if ( v55 )
    goto LABEL_44;
  if ( *(_BYTE *)(a1[26] + 3686) )
  {
LABEL_43:
    sub_324B070(a1, a3, v32, v34);
    goto LABEL_44;
  }
  v35 = a1[27];
  v36 = *(_DWORD *)(v35 + 424);
  v37 = *(_QWORD *)(v35 + 408);
  if ( !v36 )
    goto LABEL_44;
  v38 = v36 - 1;
  v39 = v38 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v40 = (__int64 *)(v37 + 16LL * v39);
  v41 = *v40;
  if ( a2 == *v40 )
  {
LABEL_42:
    if ( !v40[1] )
      goto LABEL_44;
    goto LABEL_43;
  }
  v50 = 1;
  while ( v41 != -4096 )
  {
    v51 = v50 + 1;
    v39 = v38 & (v50 + v39);
    v40 = (__int64 *)(v37 + 16LL * v39);
    v41 = *v40;
    if ( a2 == *v40 )
      goto LABEL_42;
    v50 = v51;
  }
LABEL_44:
  result = 0;
  if ( v54 )
  {
    sub_32494F0(a1, a3, 71, (unsigned __int64)v54);
    return 1;
  }
  return result;
}
