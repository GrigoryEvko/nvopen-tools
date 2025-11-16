// Function: sub_1FE5EB0
// Address: 0x1fe5eb0
//
__int64 __fastcall sub_1FE5EB0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r12
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r8
  int v10; // edx
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 (__fastcall *v13)(__int64, unsigned __int8); // r13
  unsigned int v14; // eax
  __int64 v15; // r8
  int v16; // r9d
  __int64 v17; // rsi
  __int64 v18; // rsi
  unsigned int v19; // eax
  unsigned int v20; // esi
  unsigned int v21; // r13d
  __int64 v22; // rdi
  unsigned int v23; // ecx
  unsigned __int64 *v24; // rax
  unsigned __int64 v25; // rdx
  int v26; // r9d
  int v27; // r10d
  unsigned __int64 *v28; // r9
  int v29; // edi
  int v30; // ecx
  int v31; // eax
  int v32; // esi
  __int64 v33; // r8
  unsigned int v34; // edx
  unsigned __int64 v35; // rdi
  int v36; // r10d
  unsigned __int64 *v37; // r9
  int v38; // eax
  int v39; // edx
  __int64 v40; // rdi
  unsigned __int64 *v41; // r8
  unsigned int v42; // r15d
  int v43; // r9d
  unsigned __int64 v44; // rsi

  v2 = a2 | 4;
  v4 = *(unsigned int *)(a1 + 168);
  if ( (_DWORD)v4 )
  {
    v5 = *(_QWORD *)(a1 + 152);
    v6 = (v4 - 1) & (v2 ^ (v2 >> 9));
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( v2 == *v7 )
    {
LABEL_3:
      if ( v7 != (__int64 *)(v5 + 16 * v4) )
        return *((unsigned int *)v7 + 2);
    }
    else
    {
      v10 = 1;
      while ( v8 != -4 )
      {
        v26 = v10 + 1;
        v6 = (v4 - 1) & (v10 + v6);
        v7 = (__int64 *)(v5 + 16LL * v6);
        v8 = *v7;
        if ( v2 == *v7 )
          goto LABEL_3;
        v10 = v26;
      }
    }
  }
  v11 = sub_1E0A0C0(*(_QWORD *)(a1 + 8));
  v12 = *(_QWORD *)(a1 + 16);
  v13 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v12 + 288LL);
  v14 = 8 * sub_15A9520(v11, 0);
  if ( v14 == 32 )
  {
    v17 = 5;
  }
  else if ( v14 > 0x20 )
  {
    v17 = 6;
    if ( v14 != 64 )
    {
      v17 = 0;
      if ( v14 == 128 )
        v17 = 7;
    }
  }
  else
  {
    v17 = 3;
    if ( v14 != 8 )
    {
      LOBYTE(v17) = v14 == 16;
      v17 = (unsigned int)(4 * v17);
    }
  }
  if ( v13 == sub_1D45FB0 )
    v18 = *(_QWORD *)(v12 + 8 * (v17 & 7) + 120);
  else
    v18 = v13(v12, v17);
  v19 = sub_1E6B9A0(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 40LL), v18, (unsigned __int8 *)byte_3F871B3, 0, v15, v16);
  v20 = *(_DWORD *)(a1 + 168);
  v21 = v19;
  if ( !v20 )
  {
    ++*(_QWORD *)(a1 + 144);
    goto LABEL_34;
  }
  v22 = *(_QWORD *)(a1 + 152);
  v23 = (v20 - 1) & (v2 ^ (v2 >> 9));
  v24 = (unsigned __int64 *)(v22 + 16LL * v23);
  v25 = *v24;
  if ( v2 != *v24 )
  {
    v27 = 1;
    v28 = 0;
    while ( v25 != -4 )
    {
      if ( !v28 && v25 == -16 )
        v28 = v24;
      v23 = (v20 - 1) & (v27 + v23);
      v24 = (unsigned __int64 *)(v22 + 16LL * v23);
      v25 = *v24;
      if ( v2 == *v24 )
        goto LABEL_15;
      ++v27;
    }
    v29 = *(_DWORD *)(a1 + 160);
    if ( v28 )
      v24 = v28;
    ++*(_QWORD *)(a1 + 144);
    v30 = v29 + 1;
    if ( 4 * (v29 + 1) < 3 * v20 )
    {
      if ( v20 - *(_DWORD *)(a1 + 164) - v30 > v20 >> 3 )
      {
LABEL_30:
        *(_DWORD *)(a1 + 160) = v30;
        if ( *v24 != -4 )
          --*(_DWORD *)(a1 + 164);
        *v24 = v2;
        *((_DWORD *)v24 + 2) = 0;
        goto LABEL_15;
      }
      sub_1FE5CF0(a1 + 144, v20);
      v38 = *(_DWORD *)(a1 + 168);
      if ( v38 )
      {
        v39 = v38 - 1;
        v40 = *(_QWORD *)(a1 + 152);
        v41 = 0;
        v42 = (v38 - 1) & (v2 ^ (v2 >> 9));
        v43 = 1;
        v30 = *(_DWORD *)(a1 + 160) + 1;
        v24 = (unsigned __int64 *)(v40 + 16LL * v42);
        v44 = *v24;
        if ( v2 != *v24 )
        {
          while ( v44 != -4 )
          {
            if ( !v41 && v44 == -16 )
              v41 = v24;
            v42 = v39 & (v43 + v42);
            v24 = (unsigned __int64 *)(v40 + 16LL * v42);
            v44 = *v24;
            if ( v2 == *v24 )
              goto LABEL_30;
            ++v43;
          }
          if ( v41 )
            v24 = v41;
        }
        goto LABEL_30;
      }
LABEL_62:
      ++*(_DWORD *)(a1 + 160);
      BUG();
    }
LABEL_34:
    sub_1FE5CF0(a1 + 144, 2 * v20);
    v31 = *(_DWORD *)(a1 + 168);
    if ( v31 )
    {
      v32 = v31 - 1;
      v33 = *(_QWORD *)(a1 + 152);
      v30 = *(_DWORD *)(a1 + 160) + 1;
      v34 = (v31 - 1) & (v2 ^ (v2 >> 9));
      v24 = (unsigned __int64 *)(v33 + 16LL * v34);
      v35 = *v24;
      if ( v2 != *v24 )
      {
        v36 = 1;
        v37 = 0;
        while ( v35 != -4 )
        {
          if ( v35 == -16 && !v37 )
            v37 = v24;
          v34 = v32 & (v36 + v34);
          v24 = (unsigned __int64 *)(v33 + 16LL * v34);
          v35 = *v24;
          if ( v2 == *v24 )
            goto LABEL_30;
          ++v36;
        }
        if ( v37 )
          v24 = v37;
      }
      goto LABEL_30;
    }
    goto LABEL_62;
  }
LABEL_15:
  *((_DWORD *)v24 + 2) = v21;
  return v21 | 0x100000000LL;
}
