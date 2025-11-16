// Function: sub_1552170
// Address: 0x1552170
//
void __fastcall sub_1552170(__int64 *a1, const char *a2)
{
  const char *v2; // r12
  __int64 v4; // rdi
  void (*v5)(); // rax
  __int64 v6; // rdi
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  __int64 v9; // rdi
  _DWORD *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned __int8 v13; // cl
  unsigned __int64 v14; // rsi
  unsigned __int64 v15; // rcx
  __int64 v16; // rdx
  _QWORD *v17; // rax
  __int64 *v18; // rsi
  __int64 v19; // rdi
  _WORD *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  unsigned __int8 v23; // cl
  unsigned __int64 v24; // rsi
  unsigned __int64 v25; // rcx
  __int64 v26; // rdx
  _QWORD *v27; // rax
  __int64 v28; // rdi
  _BYTE *v29; // rax
  unsigned __int64 v30; // rdx
  unsigned __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rax

  v2 = a2;
  if ( a2[16] != 78 )
    goto LABEL_2;
  v6 = *((_QWORD *)a2 - 3);
  if ( *(_BYTE *)(v6 + 16) )
    goto LABEL_2;
  if ( (*(_BYTE *)(v6 + 33) & 0x20) == 0 || *(_DWORD *)(v6 + 36) != 76 )
    goto LABEL_7;
  v9 = *a1;
  v10 = *(_DWORD **)(*a1 + 24);
  if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v10 <= 3u )
  {
    a2 = " ; (";
    sub_16E7EE0(v9, " ; (", 4);
  }
  else
  {
    *v10 = 673200928;
    *(_QWORD *)(v9 + 24) += 4LL;
  }
  v11 = *((_DWORD *)v2 + 5) & 0xFFFFFFF;
  v12 = *(_QWORD *)&v2[-24 * v11];
  v13 = *(_BYTE *)(v12 + 16);
  if ( v13 == 88 )
  {
    v32 = sub_157F120(*(_QWORD *)(v12 + 40), a2, v12);
    v12 = sub_157EBA0(v32);
    v13 = *(_BYTE *)(v12 + 16);
    v11 = *((_DWORD *)v2 + 5) & 0xFFFFFFF;
  }
  if ( v13 <= 0x17u )
  {
    v14 = 0;
    goto LABEL_20;
  }
  if ( v13 == 78 )
  {
    v31 = v12 | 4;
  }
  else
  {
    v14 = 0;
    if ( v13 != 29 )
    {
LABEL_20:
      v15 = v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF);
      goto LABEL_21;
    }
    v31 = v12 & 0xFFFFFFFFFFFFFFFBLL;
  }
  v14 = v31 & 0xFFFFFFFFFFFFFFF8LL;
  v15 = (v31 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((v31 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
  if ( (v31 & 4) == 0 )
    goto LABEL_20;
LABEL_21:
  v16 = *(_QWORD *)&v2[24 * (1 - v11)];
  v17 = *(_QWORD **)(v16 + 24);
  if ( *(_DWORD *)(v16 + 32) > 0x40u )
    v17 = (_QWORD *)*v17;
  v18 = *(__int64 **)(v15 + 24LL * (unsigned int)v17);
  sub_15520E0(a1, v18, 0);
  v19 = *a1;
  v20 = *(_WORD **)(*a1 + 24);
  if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v20 <= 1u )
  {
    v18 = (__int64 *)", ";
    sub_16E7EE0(v19, ", ", 2);
  }
  else
  {
    *v20 = 8236;
    *(_QWORD *)(v19 + 24) += 2LL;
  }
  v21 = *((_DWORD *)v2 + 5) & 0xFFFFFFF;
  v22 = *(_QWORD *)&v2[-24 * v21];
  v23 = *(_BYTE *)(v22 + 16);
  if ( v23 == 88 )
  {
    v33 = sub_157F120(*(_QWORD *)(v22 + 40), v18, v22);
    v22 = sub_157EBA0(v33);
    v23 = *(_BYTE *)(v22 + 16);
    v21 = *((_DWORD *)v2 + 5) & 0xFFFFFFF;
  }
  if ( v23 <= 0x17u )
  {
    v24 = 0;
    goto LABEL_30;
  }
  if ( v23 == 78 )
  {
    v30 = v22 | 4;
  }
  else
  {
    v24 = 0;
    if ( v23 != 29 )
    {
LABEL_30:
      v25 = v24 - 24LL * (*(_DWORD *)(v24 + 20) & 0xFFFFFFF);
      goto LABEL_31;
    }
    v30 = v22 & 0xFFFFFFFFFFFFFFFBLL;
  }
  v24 = v30 & 0xFFFFFFFFFFFFFFF8LL;
  v25 = (v30 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((v30 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
  if ( (v30 & 4) == 0 )
    goto LABEL_30;
LABEL_31:
  v26 = *(_QWORD *)&v2[24 * (2 - v21)];
  v27 = *(_QWORD **)(v26 + 24);
  if ( *(_DWORD *)(v26 + 32) > 0x40u )
    v27 = (_QWORD *)*v27;
  sub_15520E0(a1, *(__int64 **)(v25 + 24LL * (unsigned int)v27), 0);
  v28 = *a1;
  v29 = *(_BYTE **)(*a1 + 24);
  if ( *(_BYTE **)(*a1 + 16) == v29 )
  {
    sub_16E7EE0(v28, ")", 1);
  }
  else
  {
    *v29 = 41;
    ++*(_QWORD *)(v28 + 24);
  }
  if ( v2[16] == 78 )
  {
    v6 = *((_QWORD *)v2 - 3);
    if ( !*(_BYTE *)(v6 + 16) )
    {
LABEL_7:
      if ( (*(_BYTE *)(v6 + 33) & 0x20) != 0 )
      {
        v7 = sub_1649960(v6);
        if ( v8 > 9 && *(_QWORD *)v7 == 0x76766E2E6D766C6CLL && *(_WORD *)(v7 + 8) == 11885 )
          nullsub_669(v2, *a1);
      }
    }
  }
LABEL_2:
  v4 = a1[29];
  if ( v4 )
  {
    v5 = *(void (**)())(*(_QWORD *)v4 + 48LL);
    if ( v5 != nullsub_526 )
      ((void (__fastcall *)(__int64, const char *, __int64))v5)(v4, v2, *a1);
  }
}
