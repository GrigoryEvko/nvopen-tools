// Function: sub_1D92720
// Address: 0x1d92720
//
char __fastcall sub_1D92720(__int64 a1, __int64 a2, __int64 *a3, unsigned __int64 a4, unsigned int a5)
{
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v11; // rdx
  __int16 v12; // ax
  __int64 v13; // rdi
  char v14; // r8
  __int64 (*v15)(); // rax
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 (__fastcall *v18)(__int64, __int64); // rax
  __int64 v19; // rax
  int v20; // eax
  __int16 v21; // dx
  __int16 v22; // si
  int v23; // eax
  __int64 v24; // rcx
  __int64 v25; // rax
  __int16 v26; // ax
  __int64 v27; // rdi
  unsigned __int8 v29; // [rsp+7h] [rbp-59h]
  unsigned __int8 v30; // [rsp+7h] [rbp-59h]
  unsigned __int8 v31; // [rsp+7h] [rbp-59h]
  int v32; // [rsp+8h] [rbp-58h]
  int v33; // [rsp+Ch] [rbp-54h]
  char v34; // [rsp+Ch] [rbp-54h]
  char v35; // [rsp+Ch] [rbp-54h]
  char v36; // [rsp+Ch] [rbp-54h]
  char v37; // [rsp+Ch] [rbp-54h]
  _QWORD v38[2]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v39; // [rsp+20h] [rbp-40h]

  *(_BYTE *)(a2 + 1) &= ~2u;
  LODWORD(v5) = *(_DWORD *)(a2 + 224);
  *(_DWORD *)(a2 + 4) = 0;
  *(_QWORD *)(a2 + 8) = 0;
  v6 = *a3;
  v7 = *(_QWORD *)a4;
  v32 = v5;
  if ( *a3 == *(_QWORD *)a4 )
    return v5;
  while ( 1 )
  {
    v11 = *(_QWORD *)(v6 + 16);
    LOWORD(v5) = *(_WORD *)v11 - 12;
    if ( (unsigned __int16)v5 <= 1u )
      goto LABEL_5;
    v12 = *(_WORD *)(v6 + 46);
    if ( (v12 & 4) != 0 || (v12 & 8) == 0 )
    {
      if ( (*(_QWORD *)(v11 + 8) & 0x80000LL) != 0 )
        goto LABEL_14;
    }
    else if ( (unsigned __int8)sub_1E15D00(v6, 0x80000, 1) )
    {
      goto LABEL_14;
    }
    v19 = *(_QWORD *)(v6 + 16);
    if ( *(_WORD *)v19 == 1 && (v11 = *(_QWORD *)(v6 + 32), (*(_BYTE *)(v11 + 64) & 0x20) != 0)
      || ((v11 = *(unsigned __int16 *)(v6 + 46), (v11 & 4) != 0) || (v11 &= 8u, !(_DWORD)v11)
        ? (v20 = *(_DWORD *)(v19 + 12) & 1)
        : (LOBYTE(v20) = sub_1E15D00(v6, 0, 1)),
          (_BYTE)v20) )
    {
LABEL_14:
      *(_BYTE *)(a2 + 1) |= 1u;
    }
    v13 = *(_QWORD *)(a1 + 544);
    v14 = 0;
    v15 = *(__int64 (**)())(*(_QWORD *)v13 + 656LL);
    if ( v15 != sub_1D918C0 )
      v14 = ((__int64 (__fastcall *)(__int64, __int64, __int64, unsigned __int64, _QWORD))v15)(v13, v6, v11, a4, 0);
    if ( (*(_BYTE *)a2 & 0x10) == 0 )
      goto LABEL_71;
    v21 = *(_WORD *)(v6 + 46);
    v22 = v21 & 4;
    if ( (v21 & 4) != 0 || (v21 & 8) == 0 )
    {
      v24 = *(_QWORD *)(*(_QWORD *)(v6 + 16) + 8LL);
      LOBYTE(v24) = (unsigned __int8)v24 >> 7;
    }
    else
    {
      v34 = v14;
      v23 = sub_1E15D00(v6, 128, 1);
      v21 = *(_WORD *)(v6 + 46);
      v14 = v34;
      LODWORD(v24) = v23;
      v22 = v21 & 4;
    }
    if ( v22 || (v21 & 8) == 0 )
    {
      v25 = (*(_QWORD *)(*(_QWORD *)(v6 + 16) + 8LL) >> 5) & 1LL;
    }
    else
    {
      v29 = v24;
      v35 = v14;
      LODWORD(v25) = sub_1E15D00(v6, 32, 1);
      v21 = *(_WORD *)(v6 + 46);
      v14 = v35;
      LODWORD(v24) = v29;
      v22 = v21 & 4;
    }
    a4 = ((unsigned int)v25 ^ 1) & (unsigned int)v24;
    if ( v22 || (v21 & 8) == 0 )
    {
      v5 = (*(_QWORD *)(*(_QWORD *)(v6 + 16) + 8LL) >> 8) & 1LL;
    }
    else
    {
      v30 = a4;
      v36 = v14;
      LOBYTE(v5) = sub_1E15D00(v6, 256, 1);
      v14 = v36;
      a4 = v30;
    }
    if ( (_BYTE)v5 != 1 && (_BYTE)a4 )
    {
      if ( !(_BYTE)a5 )
        goto LABEL_5;
      v26 = *(_WORD *)(v6 + 46);
      a4 = a5;
      if ( (v26 & 4) != 0 )
        goto LABEL_57;
    }
    else
    {
LABEL_71:
      if ( !(_BYTE)a5 )
        break;
      v26 = *(_WORD *)(v6 + 46);
      a4 = 0;
      if ( (v26 & 4) != 0 )
        goto LABEL_57;
    }
    if ( (v26 & 8) != 0 )
    {
      v31 = a4;
      v37 = v14;
      LOBYTE(v5) = sub_1E15D00(v6, 128, 1);
      v14 = v37;
      a4 = v31;
      goto LABEL_53;
    }
LABEL_57:
    LOBYTE(v5) = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v6 + 16) + 8LL) >> 7;
LABEL_53:
    if ( (_BYTE)v5 )
      goto LABEL_37;
    if ( !(_BYTE)a4 )
      break;
LABEL_5:
    if ( (*(_BYTE *)v6 & 4) != 0 )
    {
      v6 = *(_QWORD *)(v6 + 8);
      if ( v7 == v6 )
        return v5;
    }
    else
    {
      while ( (*(_BYTE *)(v6 + 46) & 8) != 0 )
        v6 = *(_QWORD *)(v6 + 8);
      v6 = *(_QWORD *)(v6 + 8);
      if ( v7 == v6 )
        return v5;
    }
  }
  if ( v14 )
  {
    LOBYTE(v5) = v32;
    if ( !v32 )
      goto LABEL_37;
  }
  else
  {
    ++*(_DWORD *)(a2 + 4);
    v33 = (*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 544) + 856LL))(*(_QWORD *)(a1 + 544), v6);
    LODWORD(v5) = sub_1F4BF20(a1 + 256, v6, 0);
    if ( (unsigned int)v5 > 1 )
    {
      LODWORD(v5) = v5 + *(_DWORD *)(a2 + 8) - 1;
      *(_DWORD *)(a2 + 8) = v5;
    }
    *(_DWORD *)(a2 + 12) += v33;
    if ( (*(_BYTE *)(a2 + 1) & 2) != 0 )
    {
LABEL_37:
      *(_BYTE *)a2 |= 0x80u;
      return v5;
    }
  }
  v16 = *(_QWORD *)(a1 + 544);
  v38[0] = 0;
  v38[1] = 0;
  v39 = 0;
  v17 = *(_QWORD *)v16;
  a4 = *(_QWORD *)(*(_QWORD *)v16 + 712LL);
  if ( (__int64 (*)())a4 != sub_1D918E0 )
  {
    if ( ((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD *))a4)(v16, v6, v38) )
      *(_BYTE *)(a2 + 1) |= 2u;
    v16 = *(_QWORD *)(a1 + 544);
    v17 = *(_QWORD *)v16;
  }
  v18 = *(__int64 (__fastcall **)(__int64, __int64))(v17 + 720);
  if ( v18 == sub_1D918F0 )
  {
    v5 = *(_QWORD *)(v6 + 16);
    if ( (*(_BYTE *)(v5 + 10) & 4) == 0 )
      goto LABEL_63;
    goto LABEL_26;
  }
  LOBYTE(v5) = v18(v16, v6);
  if ( (_BYTE)v5 )
  {
LABEL_26:
    if ( v38[0] )
      LOBYTE(v5) = j_j___libc_free_0(v38[0], v39 - v38[0]);
    goto LABEL_5;
  }
LABEL_63:
  v27 = v38[0];
  *(_BYTE *)a2 |= 0x80u;
  if ( v27 )
    LOBYTE(v5) = j_j___libc_free_0(v27, v39 - v27);
  return v5;
}
