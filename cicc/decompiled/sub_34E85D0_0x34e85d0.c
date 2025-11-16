// Function: sub_34E85D0
// Address: 0x34e85d0
//
void __fastcall sub_34E85D0(__int64 a1, __int64 a2, __int64 *a3, __int64 *a4, char a5)
{
  int v5; // eax
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v10; // rax
  int *v11; // rdx
  int v12; // eax
  __int64 v13; // rdi
  char v14; // cl
  __int64 (*v15)(); // rax
  unsigned int v16; // eax
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 (*v19)(); // r8
  __int64 (__fastcall *v20)(__int64, __int64); // rax
  __int64 v21; // rax
  int v22; // eax
  __int64 v23; // rax
  int v24; // eax
  __int64 v25; // rax
  int v26; // eax
  __int64 v27; // rax
  int v28; // eax
  __int64 v29; // rax
  int v30; // eax
  __int64 v31; // rax
  char v32; // r8
  int v33; // eax
  __int64 v34; // rax
  unsigned __int64 v35; // rdi
  char v36; // [rsp+7h] [rbp-59h]
  int v37; // [rsp+8h] [rbp-58h]
  int v38; // [rsp+Ch] [rbp-54h]
  char v39; // [rsp+Ch] [rbp-54h]
  char v40; // [rsp+Ch] [rbp-54h]
  char v41; // [rsp+Ch] [rbp-54h]
  char v42; // [rsp+Ch] [rbp-54h]
  unsigned __int64 v43[10]; // [rsp+10h] [rbp-50h] BYREF

  *(_BYTE *)(a2 + 1) &= ~2u;
  v5 = *(_DWORD *)(a2 + 224);
  *(_DWORD *)(a2 + 4) = 0;
  *(_QWORD *)(a2 + 8) = 0;
  v6 = *a3;
  v7 = *a4;
  v37 = v5;
  if ( *a3 == *a4 )
    return;
  while ( 1 )
  {
    if ( (unsigned __int16)(*(_WORD *)(v6 + 68) - 14) <= 4u )
      goto LABEL_5;
    v10 = *(_QWORD *)(v6 + 48);
    v11 = (int *)(v10 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v10 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v12 = v10 & 7;
      switch ( v12 )
      {
        case 1:
          goto LABEL_13;
        case 3:
          v21 = *((unsigned __int8 *)v11 + 4);
          if ( (_BYTE)v21 && *(_QWORD *)&v11[2 * *v11 + 4]
            || *((_BYTE *)v11 + 5) && *(_QWORD *)&v11[2 * *v11 + 4 + 2 * v21] )
          {
            goto LABEL_13;
          }
          break;
        case 2:
          goto LABEL_13;
      }
    }
    v22 = *(_DWORD *)(v6 + 44);
    if ( (v22 & 4) == 0 && (v22 & 8) != 0 )
      LOBYTE(v23) = sub_2E88A90(v6, 0x800000, 1);
    else
      v23 = (*(_QWORD *)(*(_QWORD *)(v6 + 16) + 24LL) >> 23) & 1LL;
    if ( (_BYTE)v23
      || (unsigned int)*(unsigned __int16 *)(v6 + 68) - 1 <= 1 && (*(_BYTE *)(*(_QWORD *)(v6 + 32) + 64LL) & 0x20) != 0
      || (v24 = *(_DWORD *)(v6 + 44), (v24 & 0x20000) == 0)
      && ((v24 & 4) == 0 && (v24 & 8) != 0
        ? (LOBYTE(v25) = sub_2E88A90(v6, 0x1000000000LL, 1))
        : (v25 = (*(_QWORD *)(*(_QWORD *)(v6 + 16) + 24LL) >> 36) & 1LL),
          (_BYTE)v25) )
    {
LABEL_13:
      *(_BYTE *)(a2 + 1) |= 1u;
    }
    v13 = *(_QWORD *)(a1 + 528);
    v14 = 0;
    v15 = *(__int64 (**)())(*(_QWORD *)v13 + 920LL);
    if ( v15 != sub_2DB1B30 )
      v14 = ((__int64 (__fastcall *)(__int64, __int64, int *, _QWORD))v15)(v13, v6, v11, 0);
    if ( (*(_BYTE *)a2 & 0x10) == 0 )
      goto LABEL_83;
    v26 = *(_DWORD *)(v6 + 44);
    if ( (v26 & 4) != 0 || (v26 & 8) == 0 )
    {
      v27 = (*(_QWORD *)(*(_QWORD *)(v6 + 16) + 24LL) >> 10) & 1LL;
    }
    else
    {
      v39 = v14;
      LOBYTE(v27) = sub_2E88A90(v6, 1024, 1);
      v14 = v39;
    }
    if ( !(_BYTE)v27 )
      goto LABEL_83;
    v28 = *(_DWORD *)(v6 + 44);
    if ( (v28 & 4) == 0 && (v28 & 8) != 0 )
    {
      v41 = v14;
      LOBYTE(v29) = sub_2E88A90(v6, 256, 1);
      v14 = v41;
    }
    else
    {
      v29 = (*(_QWORD *)(*(_QWORD *)(v6 + 16) + 24LL) >> 8) & 1LL;
    }
    if ( (_BYTE)v29
      || ((v30 = *(_DWORD *)(v6 + 44), (v30 & 4) == 0) && (v30 & 8) != 0
        ? (v42 = v14, LOBYTE(v31) = sub_2E88A90(v6, 2048, 1), v14 = v42)
        : (v31 = (*(_QWORD *)(*(_QWORD *)(v6 + 16) + 24LL) >> 11) & 1LL),
          (_BYTE)v31) )
    {
LABEL_83:
      if ( !a5 )
        break;
      v32 = 0;
    }
    else
    {
      if ( !a5 )
        goto LABEL_5;
      v32 = a5;
    }
    v33 = *(_DWORD *)(v6 + 44);
    if ( (v33 & 4) != 0 || (v33 & 8) == 0 )
    {
      v34 = (*(_QWORD *)(*(_QWORD *)(v6 + 16) + 24LL) >> 10) & 1LL;
    }
    else
    {
      v40 = v14;
      v36 = v32;
      LOBYTE(v34) = sub_2E88A90(v6, 1024, 1);
      v14 = v40;
      v32 = v36;
    }
    if ( (_BYTE)v34 )
      goto LABEL_45;
    if ( !v32 )
      break;
LABEL_5:
    if ( (*(_BYTE *)v6 & 4) != 0 )
    {
      v6 = *(_QWORD *)(v6 + 8);
      if ( v7 == v6 )
        return;
    }
    else
    {
      while ( (*(_BYTE *)(v6 + 44) & 8) != 0 )
        v6 = *(_QWORD *)(v6 + 8);
      v6 = *(_QWORD *)(v6 + 8);
      if ( v7 == v6 )
        return;
    }
  }
  if ( v14 )
  {
    if ( !v37 )
      goto LABEL_45;
  }
  else
  {
    ++*(_DWORD *)(a2 + 4);
    v38 = (*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 528) + 1168LL))(*(_QWORD *)(a1 + 528), v6);
    v16 = sub_2FF8080(a1 + 224, v6, 0);
    if ( v16 > 1 )
      *(_DWORD *)(a2 + 8) = v16 + *(_DWORD *)(a2 + 8) - 1;
    *(_DWORD *)(a2 + 12) += v38;
    if ( (*(_BYTE *)(a2 + 1) & 2) != 0 )
    {
LABEL_45:
      *(_BYTE *)a2 |= 0x80u;
      return;
    }
  }
  v17 = *(_QWORD *)(a1 + 528);
  memset(v43, 0, 24);
  v18 = *(_QWORD *)v17;
  v19 = *(__int64 (**)())(*(_QWORD *)v17 + 984LL);
  if ( v19 != sub_2FDC730 )
  {
    if ( ((unsigned __int8 (__fastcall *)(__int64, __int64, unsigned __int64 *, __int64))v19)(v17, v6, v43, 1) )
      *(_BYTE *)(a2 + 1) |= 2u;
    v17 = *(_QWORD *)(a1 + 528);
    v18 = *(_QWORD *)v17;
  }
  v20 = *(__int64 (__fastcall **)(__int64, __int64))(v18 + 992);
  if ( v20 == sub_2DB1B50 )
  {
    if ( (*(_BYTE *)(*(_QWORD *)(v6 + 16) + 26LL) & 0x40) == 0 )
      goto LABEL_72;
    goto LABEL_25;
  }
  if ( (unsigned __int8)v20(v17, v6) )
  {
LABEL_25:
    if ( v43[0] )
      j_j___libc_free_0(v43[0]);
    goto LABEL_5;
  }
LABEL_72:
  v35 = v43[0];
  *(_BYTE *)a2 |= 0x80u;
  if ( v35 )
    j_j___libc_free_0(v35);
}
