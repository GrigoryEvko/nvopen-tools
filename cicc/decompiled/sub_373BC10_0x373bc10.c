// Function: sub_373BC10
// Address: 0x373bc10
//
__int64 __fastcall sub_373BC10(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  __int64 v6; // rax
  unsigned int v7; // esi
  __int64 v8; // r14
  __int64 v9; // rdi
  int v10; // r11d
  unsigned __int8 **v11; // r9
  unsigned int v12; // ecx
  _QWORD *v13; // r12
  unsigned __int8 *v14; // rax
  __int64 *v15; // r12
  __int64 result; // rax
  int v17; // eax
  int v18; // edx
  __int64 v19; // rsi
  unsigned int v20; // eax
  int v21; // ecx
  unsigned __int8 *v22; // rdi
  int v23; // r10d
  unsigned __int8 **v24; // r8
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rdi
  int v28; // eax
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rdi
  int v32; // eax
  int v33; // eax
  __int64 v34; // rdi
  int v35; // r10d
  unsigned int v36; // edx
  unsigned __int8 *v37; // rsi
  unsigned int v38; // [rsp+Ch] [rbp-34h]

  if ( !sub_3734FE0(a1) || (v8 = a1 + 704, (unsigned __int8)sub_321F6A0(*(_QWORD *)(a1 + 208), a2)) )
  {
    v6 = *(_QWORD *)(a1 + 216);
    v7 = *(_DWORD *)(v6 + 456);
    v8 = v6 + 432;
    if ( !v7 )
    {
LABEL_10:
      ++*(_QWORD *)v8;
      goto LABEL_11;
    }
  }
  else
  {
    v7 = *(_DWORD *)(a1 + 728);
    if ( !v7 )
      goto LABEL_10;
  }
  v9 = *(_QWORD *)(v8 + 8);
  v10 = 1;
  v11 = 0;
  v12 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v13 = (_QWORD *)(v9 + 16LL * v12);
  v14 = (unsigned __int8 *)*v13;
  if ( (unsigned __int8 *)*v13 == a2 )
  {
LABEL_4:
    v15 = v13 + 1;
    goto LABEL_5;
  }
  while ( v14 != (unsigned __int8 *)-4096LL )
  {
    if ( !v11 && v14 == (unsigned __int8 *)-8192LL )
      v11 = (unsigned __int8 **)v13;
    v12 = (v7 - 1) & (v10 + v12);
    v13 = (_QWORD *)(v9 + 16LL * v12);
    v14 = (unsigned __int8 *)*v13;
    if ( (unsigned __int8 *)*v13 == a2 )
      goto LABEL_4;
    ++v10;
  }
  v28 = *(_DWORD *)(v8 + 16);
  if ( !v11 )
    v11 = (unsigned __int8 **)v13;
  ++*(_QWORD *)v8;
  v21 = v28 + 1;
  if ( 4 * (v28 + 1) >= 3 * v7 )
  {
LABEL_11:
    sub_373BA10(v8, 2 * v7);
    v17 = *(_DWORD *)(v8 + 24);
    if ( v17 )
    {
      v18 = v17 - 1;
      v19 = *(_QWORD *)(v8 + 8);
      v20 = (v17 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v21 = *(_DWORD *)(v8 + 16) + 1;
      v11 = (unsigned __int8 **)(v19 + 16LL * v20);
      v22 = *v11;
      if ( *v11 != a2 )
      {
        v23 = 1;
        v24 = 0;
        while ( v22 != (unsigned __int8 *)-4096LL )
        {
          if ( !v24 && v22 == (unsigned __int8 *)-8192LL )
            v24 = v11;
          v20 = v18 & (v23 + v20);
          v11 = (unsigned __int8 **)(v19 + 16LL * v20);
          v22 = *v11;
          if ( *v11 == a2 )
            goto LABEL_33;
          ++v23;
        }
LABEL_15:
        if ( v24 )
          v11 = v24;
        goto LABEL_33;
      }
      goto LABEL_33;
    }
LABEL_54:
    ++*(_DWORD *)(v8 + 16);
    BUG();
  }
  if ( v7 - *(_DWORD *)(v8 + 20) - v21 > v7 >> 3 )
    goto LABEL_33;
  v38 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  sub_373BA10(v8, v7);
  v32 = *(_DWORD *)(v8 + 24);
  if ( !v32 )
    goto LABEL_54;
  v33 = v32 - 1;
  v34 = *(_QWORD *)(v8 + 8);
  v35 = 1;
  v24 = 0;
  v36 = v33 & v38;
  v21 = *(_DWORD *)(v8 + 16) + 1;
  v11 = (unsigned __int8 **)(v34 + 16LL * (v33 & v38));
  v37 = *v11;
  if ( *v11 != a2 )
  {
    while ( v37 != (unsigned __int8 *)-4096LL )
    {
      if ( v37 == (unsigned __int8 *)-8192LL && !v24 )
        v24 = v11;
      v36 = v33 & (v35 + v36);
      v11 = (unsigned __int8 **)(v34 + 16LL * v36);
      v37 = *v11;
      if ( *v11 == a2 )
        goto LABEL_33;
      ++v35;
    }
    goto LABEL_15;
  }
LABEL_33:
  *(_DWORD *)(v8 + 16) = v21;
  if ( *v11 != (unsigned __int8 *)-4096LL )
    --*(_DWORD *)(v8 + 20);
  *v11 = a2;
  v15 = (__int64 *)(v11 + 1);
  v11[1] = 0;
LABEL_5:
  result = *a2;
  if ( (_BYTE)result == 26 )
  {
    v25 = sub_22077B0(0x60u);
    v26 = v25;
    if ( v25 )
    {
      *(_QWORD *)(v25 + 8) = a2;
      *(_QWORD *)(v25 + 16) = 0;
      *(_QWORD *)(v25 + 24) = 0;
      *(_DWORD *)(v25 + 32) = 0;
      *(_QWORD *)v25 = &unk_4A35790;
      *(_WORD *)(v25 + 88) = 0;
    }
    v27 = *v15;
    *v15 = v25;
    if ( v27 )
    {
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v27 + 8LL))(v27);
      v26 = *v15;
    }
    return (__int64)sub_3245B60(*(_QWORD *)(a1 + 216), a3, v26);
  }
  else if ( (_BYTE)result == 27 )
  {
    v29 = sub_22077B0(0x30u);
    v30 = v29;
    if ( v29 )
    {
      *(_QWORD *)(v29 + 8) = a2;
      *(_QWORD *)(v29 + 16) = 0;
      *(_QWORD *)(v29 + 24) = 0;
      *(_DWORD *)(v29 + 32) = 1;
      *(_QWORD *)(v29 + 40) = 0;
      *(_QWORD *)v29 = &unk_4A357B0;
    }
    v31 = *v15;
    *v15 = v29;
    if ( v31 )
    {
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v31 + 8LL))(v31);
      v30 = *v15;
    }
    return sub_3246200(*(_QWORD *)(a1 + 216), a3, v30);
  }
  return result;
}
