// Function: sub_31C4220
// Address: 0x31c4220
//
_QWORD *__fastcall sub_31C4220(__int64 a1, __int64 a2)
{
  __int64 v4; // r13
  __int64 v5; // rax
  _DWORD *v6; // rdi
  unsigned int v7; // esi
  __int64 v8; // r13
  __int64 v9; // rdi
  int v10; // r9d
  __int64 *v11; // r11
  unsigned int v12; // ecx
  __int64 *v13; // rax
  __int64 v14; // rdx
  _QWORD *result; // rax
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // r12
  __int64 v21; // rdx
  int v22; // eax
  __int64 *v23; // rdx
  int v24; // eax
  int v25; // edx
  _QWORD *v26; // r14
  __int64 *v27; // rax
  __int32 v28; // eax
  int v29; // eax
  int v30; // esi
  __int64 v31; // rdi
  unsigned int v32; // ecx
  __int64 v33; // rax
  int v34; // r10d
  __int64 *v35; // r8
  int v36; // eax
  int v37; // ecx
  __int64 v38; // rsi
  int v39; // r8d
  unsigned int v40; // r12d
  __int64 *v41; // rdi
  __int64 v42; // rax
  __int32 v43; // [rsp+8h] [rbp-58h]
  __m128i v44[5]; // [rsp+10h] [rbp-50h] BYREF

  sub_31C2BE0((__int64)v44, a1, a2);
  v4 = sub_31C3EB0(a1, v44);
  v5 = *(unsigned int *)(v4 + 8);
  if ( (_DWORD)v5 )
  {
    v6 = *(_DWORD **)(*(_QWORD *)v4 + 8 * v5 - 8);
    if ( v6[4] != (_DWORD)qword_5035D08 )
    {
      (*(void (__fastcall **)(_DWORD *, __int64, _QWORD))(*(_QWORD *)v6 + 16LL))(v6, a2, *(_QWORD *)(a1 + 80));
      goto LABEL_4;
    }
  }
  v16 = sub_22077B0(0x98u);
  v20 = v16;
  if ( v16 )
  {
    *(_QWORD *)(v16 + 72) = v16 + 88;
    *(_QWORD *)(v16 + 8) = v16 + 24;
    *(_QWORD *)v16 = &unk_4A34AB0;
    *(_QWORD *)(v16 + 16) = 0x600000000LL;
    *(_QWORD *)(v16 + 80) = 0x600000000LL;
    *(_DWORD *)(v16 + 136) = 0;
    *(_QWORD *)(v16 + 144) = 0;
    sub_31C2860(v16, (char *)(v16 + 24), a2, v17, v18, v19);
    *(_QWORD *)v20 = &unk_4A34AD8;
  }
  v21 = *(unsigned int *)(v4 + 8);
  v22 = v21;
  if ( *(_DWORD *)(v4 + 12) <= (unsigned int)v21 )
  {
    v26 = (_QWORD *)sub_C8D7D0(v4, v4 + 16, 0, 8u, (unsigned __int64 *)v44, v19);
    v27 = &v26[*(unsigned int *)(v4 + 8)];
    if ( v27 )
    {
      *v27 = v20;
      v20 = 0;
    }
    sub_31C3490(v4, v26);
    v28 = v44[0].m128i_i32[0];
    if ( v4 + 16 != *(_QWORD *)v4 )
    {
      v43 = v44[0].m128i_i32[0];
      _libc_free(*(_QWORD *)v4);
      v28 = v43;
    }
    ++*(_DWORD *)(v4 + 8);
    *(_QWORD *)v4 = v26;
    *(_DWORD *)(v4 + 12) = v28;
  }
  else
  {
    v23 = (__int64 *)(*(_QWORD *)v4 + 8 * v21);
    if ( v23 )
    {
      *v23 = v20;
      ++*(_DWORD *)(v4 + 8);
      goto LABEL_4;
    }
    *(_DWORD *)(v4 + 8) = v22 + 1;
  }
  if ( v20 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v20 + 8LL))(v20);
LABEL_4:
  v7 = *(_DWORD *)(a1 + 72);
  v8 = *(_QWORD *)(*(_QWORD *)v4 + 8LL * *(unsigned int *)(v4 + 8) - 8);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 48);
    goto LABEL_35;
  }
  v9 = *(_QWORD *)(a1 + 56);
  v10 = 1;
  v11 = 0;
  v12 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v13 = (__int64 *)(v9 + 16LL * v12);
  v14 = *v13;
  if ( *v13 == a2 )
  {
LABEL_6:
    result = v13 + 1;
    goto LABEL_7;
  }
  while ( v14 != -4096 )
  {
    if ( !v11 && v14 == -8192 )
      v11 = v13;
    v12 = (v7 - 1) & (v10 + v12);
    v13 = (__int64 *)(v9 + 16LL * v12);
    v14 = *v13;
    if ( *v13 == a2 )
      goto LABEL_6;
    ++v10;
  }
  if ( !v11 )
    v11 = v13;
  v24 = *(_DWORD *)(a1 + 64);
  ++*(_QWORD *)(a1 + 48);
  v25 = v24 + 1;
  if ( 4 * (v24 + 1) >= 3 * v7 )
  {
LABEL_35:
    sub_31C3AD0(a1 + 48, 2 * v7);
    v29 = *(_DWORD *)(a1 + 72);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a1 + 56);
      v32 = (v29 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v25 = *(_DWORD *)(a1 + 64) + 1;
      v11 = (__int64 *)(v31 + 16LL * v32);
      v33 = *v11;
      if ( *v11 != a2 )
      {
        v34 = 1;
        v35 = 0;
        while ( v33 != -4096 )
        {
          if ( !v35 && v33 == -8192 )
            v35 = v11;
          v32 = v30 & (v34 + v32);
          v11 = (__int64 *)(v31 + 16LL * v32);
          v33 = *v11;
          if ( *v11 == a2 )
            goto LABEL_26;
          ++v34;
        }
        if ( v35 )
          v11 = v35;
      }
      goto LABEL_26;
    }
    goto LABEL_58;
  }
  if ( v7 - *(_DWORD *)(a1 + 68) - v25 <= v7 >> 3 )
  {
    sub_31C3AD0(a1 + 48, v7);
    v36 = *(_DWORD *)(a1 + 72);
    if ( v36 )
    {
      v37 = v36 - 1;
      v38 = *(_QWORD *)(a1 + 56);
      v39 = 1;
      v40 = (v36 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v25 = *(_DWORD *)(a1 + 64) + 1;
      v41 = 0;
      v11 = (__int64 *)(v38 + 16LL * v40);
      v42 = *v11;
      if ( *v11 != a2 )
      {
        while ( v42 != -4096 )
        {
          if ( !v41 && v42 == -8192 )
            v41 = v11;
          v40 = v37 & (v39 + v40);
          v11 = (__int64 *)(v38 + 16LL * v40);
          v42 = *v11;
          if ( *v11 == a2 )
            goto LABEL_26;
          ++v39;
        }
        if ( v41 )
          v11 = v41;
      }
      goto LABEL_26;
    }
LABEL_58:
    ++*(_DWORD *)(a1 + 64);
    BUG();
  }
LABEL_26:
  *(_DWORD *)(a1 + 64) = v25;
  if ( *v11 != -4096 )
    --*(_DWORD *)(a1 + 68);
  *v11 = a2;
  result = v11 + 1;
  v11[1] = 0;
LABEL_7:
  *result = v8;
  return result;
}
