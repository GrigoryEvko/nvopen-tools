// Function: sub_1E694D0
// Address: 0x1e694d0
//
__int64 __fastcall sub_1E694D0(__int64 a1, int a2, int a3, unsigned int a4)
{
  unsigned int v5; // r8d
  __int64 v6; // r10
  unsigned int v7; // r9d
  __int64 v10; // r11
  __int64 v12; // rdx
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rcx
  unsigned __int64 v16; // rbx
  __int64 v17; // r12
  __int64 v18; // rdi
  __int64 (*v19)(); // rax
  __int64 v20; // rdx
  __int64 v21; // r9
  bool v22; // di
  unsigned __int8 v23; // r11
  __int64 v24; // rax
  __int64 v25; // r9
  bool v26; // r8
  unsigned __int8 v27; // r10
  __int64 v28; // rax
  unsigned __int64 v29; // rsi
  unsigned __int64 v30; // rdx
  _BYTE *v32; // rax
  _BYTE *v33; // rdi
  unsigned __int64 v34; // [rsp+8h] [rbp-38h]

  v5 = a2 & 0x7FFFFFFF;
  v6 = a2 & 0x7FFFFFFF;
  v7 = a3 & 0x7FFFFFFF;
  v10 = a3 & 0x7FFFFFFF;
  v12 = *(_QWORD *)(a1 + 24);
  v13 = *(_QWORD *)(v12 + 16 * v6);
  v14 = *(_QWORD *)(v12 + 16 * v10);
  v15 = (v13 >> 2) & 1;
  if ( ((v13 >> 2) & 1) != 0 )
  {
    v17 = (v14 >> 2) & 1;
    if ( ((v14 >> 2) & 1) == 0 && (v14 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      goto LABEL_23;
    goto LABEL_11;
  }
  v16 = v13 & 0xFFFFFFFFFFFFFFF8LL;
  v17 = (v14 >> 2) & 1;
  if ( ((v14 >> 2) & 1) != 0 )
  {
    if ( v16 )
      goto LABEL_23;
    if ( a3 >= 0 )
      goto LABEL_12;
LABEL_27:
    if ( *(_DWORD *)(a1 + 336) > v7 )
    {
      v32 = (_BYTE *)(*(_QWORD *)(a1 + 328) + 8 * v10);
      v23 = *v32 & 1;
      v22 = (*v32 & 2) != 0;
      v21 = *(_QWORD *)v32 >> 2;
      goto LABEL_13;
    }
LABEL_12:
    v21 = 0;
    v22 = 0;
    v23 = 0;
LABEL_13:
    v24 = (4 * v21) | v23 | (2LL * v22);
    if ( a2 >= 0 || v5 >= *(_DWORD *)(a1 + 336) )
    {
      v25 = 0;
      v26 = 0;
      v27 = 0;
    }
    else
    {
      v33 = (_BYTE *)(*(_QWORD *)(a1 + 328) + 8 * v6);
      v27 = *v33 & 1;
      v26 = (*v33 & 2) != 0;
      v25 = *(_QWORD *)v33 >> 2;
    }
    if ( (((unsigned __int8)v24 ^ (unsigned __int8)((4 * v25) | v27 | (2 * v26))) & 3) != 0 )
      goto LABEL_23;
    v28 = ((4 * v25) | v27 | (2LL * v26)) ^ v24;
    if ( (v28 & 0xFFFFFFFFFFFFFFFCLL) != 0 )
      goto LABEL_23;
    if ( (_BYTE)v15 )
    {
      v29 = v13 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !(_BYTE)v17 )
      {
        LODWORD(v17) = v15;
        return (unsigned int)v17;
      }
      v30 = v14 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v29 )
      {
        LOBYTE(v17) = v30 == v29;
        LOBYTE(v28) = v30 == 0;
        LODWORD(v17) = v28 | v17;
        return (unsigned int)v17;
      }
    }
    else
    {
      v30 = v14 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !(_BYTE)v17 )
        goto LABEL_36;
    }
    if ( v30 )
    {
      LODWORD(v17) = 1;
      sub_1E693F0(a1, a2, v30);
      return (unsigned int)v17;
    }
LABEL_36:
    LODWORD(v17) = 1;
    return (unsigned int)v17;
  }
  LODWORD(v17) = v14 & 0xFFFFFFF8;
  v34 = v14 & 0xFFFFFFFFFFFFFFF8LL;
  LOBYTE(v17) = v16 != 0 && (v14 & 0xFFFFFFFFFFFFFFF8LL) != 0;
  if ( !(_BYTE)v17 )
  {
    if ( ((v13 | v14) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      goto LABEL_23;
LABEL_11:
    if ( a3 >= 0 )
      goto LABEL_12;
    goto LABEL_27;
  }
  if ( v34 != v16 )
  {
    v18 = 0;
    v19 = *(__int64 (**)())(**(_QWORD **)(*(_QWORD *)a1 + 16LL) + 112LL);
    if ( v19 != sub_1D00B10 )
      v18 = ((__int64 (__fastcall *)(_QWORD))v19)(*(_QWORD *)(*(_QWORD *)a1 + 16LL));
    v20 = sub_1F4AF90(v18, v13 & 0xFFFFFFFFFFFFFFF8LL, v34, 255);
    if ( v16 == v20 || v20 == 0 )
    {
      LOBYTE(v17) = v20 != 0;
      return (unsigned int)v17;
    }
    if ( a4 <= *(unsigned __int16 *)(*(_QWORD *)v20 + 20LL) )
    {
      sub_1E693D0(a1, a2, v20);
      return (unsigned int)v17;
    }
LABEL_23:
    LODWORD(v17) = 0;
  }
  return (unsigned int)v17;
}
