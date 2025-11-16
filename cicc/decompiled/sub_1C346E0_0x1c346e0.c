// Function: sub_1C346E0
// Address: 0x1c346e0
//
__int64 __fastcall sub_1C346E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  unsigned int v7; // esi
  __int64 v8; // r8
  unsigned int v9; // ecx
  __int64 *v10; // rdx
  __int64 result; // rax
  int v12; // r10d
  __int64 *v13; // r9
  int v14; // eax
  int v15; // edx
  char v16; // al
  int v17; // eax
  int v18; // ecx
  __int64 v19; // rdi
  unsigned int v20; // eax
  __int64 v21; // rsi
  int v22; // r10d
  __int64 *v23; // r8
  __int64 *v24; // r13
  __int64 *v25; // r15
  __int64 v26; // rsi
  unsigned __int8 v27; // al
  __int64 v28; // rax
  int v29; // eax
  int v30; // eax
  __int64 v31; // rsi
  int v32; // r8d
  unsigned int v33; // r13d
  __int64 *v34; // rdi
  __int64 v35; // rcx
  __int64 v36; // rax
  char *v37; // [rsp+0h] [rbp-50h] BYREF
  size_t v38; // [rsp+8h] [rbp-48h]
  _QWORD v39[8]; // [rsp+10h] [rbp-40h] BYREF

  v3 = a1 + 112;
  v7 = *(_DWORD *)(a1 + 136);
  if ( !v7 )
    goto LABEL_14;
LABEL_2:
  v8 = *(_QWORD *)(a1 + 120);
  v9 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v8 + 8LL * v9);
  result = *v10;
  if ( *v10 != a2 )
  {
    v12 = 1;
    v13 = 0;
    while ( result != -8 )
    {
      if ( v13 || result != -16 )
        v10 = v13;
      v9 = (v7 - 1) & (v12 + v9);
      result = *(_QWORD *)(v8 + 8LL * v9);
      if ( result == a2 )
        return result;
      ++v12;
      v13 = v10;
      v10 = (__int64 *)(v8 + 8LL * v9);
    }
    v14 = *(_DWORD *)(a1 + 128);
    if ( !v13 )
      v13 = v10;
    ++*(_QWORD *)(a1 + 112);
    v15 = v14 + 1;
    if ( 4 * (v14 + 1) < 3 * v7 )
    {
      if ( v7 - *(_DWORD *)(a1 + 132) - v15 > v7 >> 3 )
        goto LABEL_10;
      sub_1C34530(v3, v7);
      v29 = *(_DWORD *)(a1 + 136);
      if ( v29 )
      {
        v30 = v29 - 1;
        v31 = *(_QWORD *)(a1 + 120);
        v32 = 1;
        v33 = v30 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v13 = (__int64 *)(v31 + 8LL * v33);
        v15 = *(_DWORD *)(a1 + 128) + 1;
        v34 = 0;
        v35 = *v13;
        if ( *v13 == a2 )
          goto LABEL_10;
        while ( 1 )
        {
          if ( v35 == -8 )
          {
            if ( v34 )
              v13 = v34;
            goto LABEL_10;
          }
          if ( v35 == -16 && !v34 )
            v34 = v13;
          v33 = v30 & (v32 + v33);
          v13 = (__int64 *)(v31 + 8LL * v33);
          v35 = *v13;
          if ( *v13 == a2 )
            goto LABEL_10;
          ++v32;
        }
      }
LABEL_65:
      ++*(_DWORD *)(a1 + 128);
      BUG();
    }
    while ( 1 )
    {
      sub_1C34530(v3, 2 * v7);
      v17 = *(_DWORD *)(a1 + 136);
      if ( !v17 )
        goto LABEL_65;
      v18 = v17 - 1;
      v19 = *(_QWORD *)(a1 + 120);
      v20 = (v17 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v13 = (__int64 *)(v19 + 8LL * v20);
      v21 = *v13;
      v15 = *(_DWORD *)(a1 + 128) + 1;
      if ( *v13 != a2 )
      {
        v22 = 1;
        v23 = 0;
        while ( v21 != -8 )
        {
          if ( !v23 && v21 == -16 )
            v23 = v13;
          v20 = v18 & (v22 + v20);
          v13 = (__int64 *)(v19 + 8LL * v20);
          v21 = *v13;
          if ( *v13 == a2 )
            goto LABEL_10;
          ++v22;
        }
        if ( v23 )
          v13 = v23;
      }
LABEL_10:
      *(_DWORD *)(a1 + 128) = v15;
      if ( *v13 != -8 )
        --*(_DWORD *)(a1 + 132);
      *v13 = a2;
      v16 = *(_BYTE *)(a2 + 8);
      if ( v16 != 15 && ((v16 - 14) & 0xFD) != 0 )
        break;
      v7 = *(_DWORD *)(a1 + 136);
      a2 = *(_QWORD *)(a2 + 24);
      if ( v7 )
        goto LABEL_2;
LABEL_14:
      ++*(_QWORD *)(a1 + 112);
    }
    if ( v16 == 13 )
    {
      v24 = *(__int64 **)(a2 + 16);
      v25 = &v24[*(unsigned int *)(a2 + 12)];
      while ( v24 != v25 )
      {
        v26 = *v24++;
        sub_1C346E0(a1, v26, a3);
      }
    }
    v38 = 0;
    LOBYTE(v39[0]) = 0;
    result = *(unsigned __int8 *)(a2 + 8);
    v37 = (char *)v39;
    if ( (_BYTE)result == 6 )
    {
      sub_2241130(&v37, 0, 0, "ppc_fp128 type is not supported\n", 32);
      result = v38;
    }
    else if ( (unsigned __int8)result > 6u )
    {
      if ( (_BYTE)result != 9 )
        return result;
      sub_2241130(&v37, 0, 0, "x86mmx type is not supported\n", 29);
      result = v38;
    }
    else if ( (_BYTE)result == 4 )
    {
      sub_2241130(&v37, 0, 0, "x86_fp80 type is not supported\n", 31);
      result = v38;
    }
    else
    {
      if ( (_BYTE)result != 5 )
        return result;
      sub_2241130(&v37, 0, 0, "fp128 type is not supported\n", 28);
      result = v38;
    }
    if ( result )
    {
      v27 = *(_BYTE *)(a3 + 16);
      if ( v27 <= 0x17u )
      {
        if ( v27 == 3 )
        {
          v36 = sub_1C31E60(a1, a3, 0);
          sub_16E7EE0(v36, v37, v38);
        }
        else
        {
          sub_1263B40(*(_QWORD *)(a1 + 24), "Error: ");
          sub_16E7EE0(*(_QWORD *)(a1 + 24), v37, v38);
        }
      }
      else
      {
        v28 = sub_1C321C0(a1, a3, 0);
        sub_16E7EE0(v28, v37, v38);
      }
      result = sub_1C31880(a1);
    }
    if ( v37 != (char *)v39 )
      return j_j___libc_free_0(v37, v39[0] + 1LL);
  }
  return result;
}
