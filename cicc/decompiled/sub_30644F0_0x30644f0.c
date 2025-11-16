// Function: sub_30644F0
// Address: 0x30644f0
//
__int64 __fastcall sub_30644F0(__int64 **a1, __int64 a2)
{
  unsigned int v3; // esi
  __int64 v4; // r8
  __int64 v5; // r14
  __int64 v6; // r9
  __int64 v7; // r15
  __int64 v8; // r10
  unsigned int v9; // edi
  __int64 result; // rax
  void *v11; // rcx
  __int64 *v12; // rbx
  _QWORD *v13; // rdx
  int v14; // eax
  int v15; // ecx
  __int64 v16; // rdi
  int v17; // r10d
  int v18; // r10d
  __int64 v19; // rsi
  unsigned int v20; // r11d
  void *v21; // rax
  int v22; // edi
  _QWORD *v23; // r13
  int v24; // r10d
  int v25; // r10d
  __int64 v26; // rsi
  _QWORD *v27; // r11
  unsigned int v28; // r13d
  int v29; // edi
  void *v30; // rax
  __int64 v31; // [rsp+8h] [rbp-58h]
  __int64 v32; // [rsp+8h] [rbp-58h]
  __int64 v33; // [rsp+8h] [rbp-58h]
  int v34; // [rsp+10h] [rbp-50h]
  __int64 v35; // [rsp+10h] [rbp-50h]
  __int64 v36; // [rsp+10h] [rbp-50h]
  __int64 v37; // [rsp+10h] [rbp-50h]
  int v38; // [rsp+1Ch] [rbp-44h]
  __int64 v39; // [rsp+20h] [rbp-40h]
  __int64 v40; // [rsp+28h] [rbp-38h]

  v3 = *(_DWORD *)(a2 + 24);
  v4 = **a1;
  v5 = (*a1)[3];
  v6 = (*a1)[4];
  v7 = (*a1)[5];
  v38 = *((_DWORD *)*a1 + 12);
  v40 = (*a1)[1];
  v39 = (*a1)[2];
  if ( !v3 )
  {
    ++*(_QWORD *)a2;
    goto LABEL_19;
  }
  v8 = *(_QWORD *)(a2 + 8);
  v9 = (v3 - 1) & (((unsigned int)&unk_5035D48 >> 9) ^ ((unsigned int)&unk_5035D48 >> 4));
  result = v8 + 16LL * v9;
  v11 = *(void **)result;
  if ( *(_UNKNOWN **)result == &unk_5035D48 )
  {
LABEL_3:
    v12 = (__int64 *)(result + 8);
    if ( *(_QWORD *)(result + 8) )
      return result;
    goto LABEL_14;
  }
  v34 = 1;
  v13 = 0;
  while ( v11 != (void *)-4096LL )
  {
    if ( !v13 && v11 == (void *)-8192LL )
      v13 = (_QWORD *)result;
    v9 = (v3 - 1) & (v34 + v9);
    result = v8 + 16LL * v9;
    v11 = *(void **)result;
    if ( *(_UNKNOWN **)result == &unk_5035D48 )
      goto LABEL_3;
    ++v34;
  }
  if ( !v13 )
    v13 = (_QWORD *)result;
  v14 = *(_DWORD *)(a2 + 16);
  ++*(_QWORD *)a2;
  v15 = v14 + 1;
  if ( 4 * (v14 + 1) >= 3 * v3 )
  {
LABEL_19:
    v32 = v6;
    v36 = v4;
    sub_2275E10(a2, 2 * v3);
    v17 = *(_DWORD *)(a2 + 24);
    if ( v17 )
    {
      v18 = v17 - 1;
      v19 = *(_QWORD *)(a2 + 8);
      v4 = v36;
      v6 = v32;
      v15 = *(_DWORD *)(a2 + 16) + 1;
      v20 = v18 & (((unsigned int)&unk_5035D48 >> 9) ^ ((unsigned int)&unk_5035D48 >> 4));
      v13 = (_QWORD *)(v19 + 16LL * v20);
      v21 = (void *)*v13;
      if ( (_UNKNOWN *)*v13 != &unk_5035D48 )
      {
        v22 = 1;
        v23 = 0;
        while ( v21 != (void *)-4096LL )
        {
          if ( !v23 && v21 == (void *)-8192LL )
            v23 = v13;
          v20 = v18 & (v22 + v20);
          v13 = (_QWORD *)(v19 + 16LL * v20);
          v21 = (void *)*v13;
          if ( (_UNKNOWN *)*v13 == &unk_5035D48 )
            goto LABEL_11;
          ++v22;
        }
        if ( v23 )
          v13 = v23;
      }
      goto LABEL_11;
    }
    goto LABEL_47;
  }
  if ( v3 - *(_DWORD *)(a2 + 20) - v15 <= v3 >> 3 )
  {
    v33 = v6;
    v37 = v4;
    sub_2275E10(a2, v3);
    v24 = *(_DWORD *)(a2 + 24);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a2 + 8);
      v27 = 0;
      v28 = v25 & (((unsigned int)&unk_5035D48 >> 9) ^ ((unsigned int)&unk_5035D48 >> 4));
      v4 = v37;
      v6 = v33;
      v29 = 1;
      v15 = *(_DWORD *)(a2 + 16) + 1;
      v13 = (_QWORD *)(v26 + 16LL * v28);
      v30 = (void *)*v13;
      if ( (_UNKNOWN *)*v13 != &unk_5035D48 )
      {
        while ( v30 != (void *)-4096LL )
        {
          if ( !v27 && v30 == (void *)-8192LL )
            v27 = v13;
          v28 = v25 & (v29 + v28);
          v13 = (_QWORD *)(v26 + 16LL * v28);
          v30 = (void *)*v13;
          if ( (_UNKNOWN *)*v13 == &unk_5035D48 )
            goto LABEL_11;
          ++v29;
        }
        if ( v27 )
          v13 = v27;
      }
      goto LABEL_11;
    }
LABEL_47:
    ++*(_DWORD *)(a2 + 16);
    BUG();
  }
LABEL_11:
  *(_DWORD *)(a2 + 16) = v15;
  if ( *v13 != -4096 )
    --*(_DWORD *)(a2 + 20);
  *v13 = &unk_5035D48;
  v12 = v13 + 1;
  v13[1] = 0;
LABEL_14:
  v31 = v6;
  v35 = v4;
  result = sub_22077B0(0x40u);
  if ( result )
  {
    *(_QWORD *)(result + 32) = v5;
    *(_QWORD *)(result + 48) = v7;
    *(_QWORD *)(result + 8) = v35;
    *(_QWORD *)result = &unk_4A30EB0;
    *(_QWORD *)(result + 40) = v31;
    *(_QWORD *)(result + 16) = v40;
    *(_DWORD *)(result + 56) = v38;
    *(_QWORD *)(result + 24) = v39;
  }
  v16 = *v12;
  *v12 = result;
  if ( v16 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v16 + 8LL))(v16);
  return result;
}
