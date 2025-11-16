// Function: sub_275C7B0
// Address: 0x275c7b0
//
__int64 __fastcall sub_275C7B0(__int64 a1, unsigned __int8 *a2)
{
  __int64 v4; // rdi
  unsigned int v5; // esi
  int v6; // r11d
  __int64 v7; // rcx
  __int64 v8; // r14
  unsigned int v9; // edx
  __int64 v10; // rax
  unsigned __int8 *v11; // r9
  __int64 result; // rax
  int v13; // eax
  int v14; // edx
  __int64 v15; // rbx
  char v16; // al
  int v17; // eax
  int v18; // ecx
  __int64 v19; // rdi
  unsigned int v20; // eax
  unsigned __int8 *v21; // rsi
  int v22; // r9d
  __int64 v23; // r8
  int v24; // eax
  int v25; // eax
  __int64 v26; // rsi
  __int64 v27; // rdi
  unsigned int v28; // r13d
  int v29; // r8d
  unsigned __int8 *v30; // rcx
  char v31; // [rsp-69h] [rbp-69h] BYREF
  unsigned __int8 *v32; // [rsp-68h] [rbp-68h] BYREF
  char v33; // [rsp-60h] [rbp-60h] BYREF
  __int64 v34; // [rsp-58h] [rbp-58h] BYREF
  __int64 v35; // [rsp-48h] [rbp-48h]
  char v36; // [rsp-38h] [rbp-38h]

  if ( *a2 == 60 )
    return 1;
  v4 = a1 + 1464;
  v5 = *(_DWORD *)(a1 + 1488);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 1464);
    goto LABEL_30;
  }
  v6 = 1;
  v7 = *(_QWORD *)(a1 + 1472);
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = v7 + 16LL * v9;
  v11 = *(unsigned __int8 **)v10;
  if ( a2 == *(unsigned __int8 **)v10 )
    return *(unsigned __int8 *)(v10 + 8);
  while ( v11 != (unsigned __int8 *)-4096LL )
  {
    if ( v11 == (unsigned __int8 *)-8192LL && !v8 )
      v8 = v10;
    v9 = (v5 - 1) & (v6 + v9);
    v10 = v7 + 16LL * v9;
    v11 = *(unsigned __int8 **)v10;
    if ( a2 == *(unsigned __int8 **)v10 )
      return *(unsigned __int8 *)(v10 + 8);
    ++v6;
  }
  if ( !v8 )
    v8 = v10;
  v13 = *(_DWORD *)(a1 + 1480);
  ++*(_QWORD *)(a1 + 1464);
  v14 = v13 + 1;
  if ( 4 * (v13 + 1) >= 3 * v5 )
  {
LABEL_30:
    sub_275AA00(v4, 2 * v5);
    v17 = *(_DWORD *)(a1 + 1488);
    if ( v17 )
    {
      v18 = v17 - 1;
      v19 = *(_QWORD *)(a1 + 1472);
      v20 = (v17 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v14 = *(_DWORD *)(a1 + 1480) + 1;
      v8 = v19 + 16LL * v20;
      v21 = *(unsigned __int8 **)v8;
      if ( a2 != *(unsigned __int8 **)v8 )
      {
        v22 = 1;
        v23 = 0;
        while ( v21 != (unsigned __int8 *)-4096LL )
        {
          if ( !v23 && v21 == (unsigned __int8 *)-8192LL )
            v23 = v8;
          v20 = v18 & (v22 + v20);
          v8 = v19 + 16LL * v20;
          v21 = *(unsigned __int8 **)v8;
          if ( a2 == *(unsigned __int8 **)v8 )
            goto LABEL_16;
          ++v22;
        }
        if ( v23 )
          v8 = v23;
      }
      goto LABEL_16;
    }
    goto LABEL_53;
  }
  if ( v5 - *(_DWORD *)(a1 + 1484) - v14 <= v5 >> 3 )
  {
    sub_275AA00(v4, v5);
    v24 = *(_DWORD *)(a1 + 1488);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a1 + 1472);
      v27 = 0;
      v28 = v25 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v29 = 1;
      v14 = *(_DWORD *)(a1 + 1480) + 1;
      v8 = v26 + 16LL * v28;
      v30 = *(unsigned __int8 **)v8;
      if ( a2 != *(unsigned __int8 **)v8 )
      {
        while ( v30 != (unsigned __int8 *)-4096LL )
        {
          if ( v30 == (unsigned __int8 *)-8192LL && !v27 )
            v27 = v8;
          v28 = v25 & (v29 + v28);
          v8 = v26 + 16LL * v28;
          v30 = *(unsigned __int8 **)v8;
          if ( a2 == *(unsigned __int8 **)v8 )
            goto LABEL_16;
          ++v29;
        }
        if ( v27 )
          v8 = v27;
      }
      goto LABEL_16;
    }
LABEL_53:
    ++*(_DWORD *)(a1 + 1480);
    BUG();
  }
LABEL_16:
  *(_DWORD *)(a1 + 1480) = v14;
  if ( *(_QWORD *)v8 != -4096 )
    --*(_DWORD *)(a1 + 1484);
  *(_QWORD *)v8 = a2;
  *(_BYTE *)(v8 + 8) = 0;
  if ( !(unsigned __int8)sub_CF7590(a2, &v31)
    || v31
    && ((v32 = a2, v33 = 1, sub_275ABE0((__int64)&v34, a1 + 1432, (__int64 *)&v32, &v33), v15 = v35, v36)
      ? (v16 = sub_D13FA0((__int64)a2, 0, 0), *(_BYTE *)(v15 + 8) = v16)
      : (v16 = *(_BYTE *)(v35 + 8)),
        v16) )
  {
    *(_BYTE *)(v8 + 8) = 0;
    return 0;
  }
  else if ( (unsigned __int8)sub_CF6FD0(a2) )
  {
    result = (unsigned int)sub_D13FA0((__int64)a2, 1, 0) ^ 1;
    *(_BYTE *)(v8 + 8) = result;
  }
  else
  {
    return *(unsigned __int8 *)(v8 + 8);
  }
  return result;
}
