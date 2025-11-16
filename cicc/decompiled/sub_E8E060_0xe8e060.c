// Function: sub_E8E060
// Address: 0xe8e060
//
unsigned __int64 __fastcall sub_E8E060(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rbx
  __int64 *(__fastcall *v10)(__int64, __int64, __int64, __int64, __int64, __int64); // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  unsigned __int64 result; // rax
  unsigned int v16; // esi
  __int64 v17; // r10
  __int64 v18; // r8
  __int64 v19; // r9
  int v20; // r11d
  unsigned int v21; // edi
  unsigned __int64 v22; // r15
  __int64 v23; // rcx
  __int64 v24; // rdx
  unsigned __int64 v25; // rbx
  __int64 *v26; // rdx
  int v27; // ecx
  int v28; // edi
  int v29; // eax
  int v30; // ecx
  __int64 v31; // r8
  unsigned int v32; // edx
  __int64 v33; // rsi
  int v34; // r10d
  unsigned __int64 v35; // r9
  int v36; // eax
  int v37; // ecx
  int v38; // r10d
  __int64 v39; // r8
  unsigned int v40; // edx
  __int64 v41; // rsi

  v9 = *(_QWORD *)(a3 + 16);
  if ( (*(_BYTE *)(v9 + 8) & 0x10) == 0 )
  {
    v16 = *(_DWORD *)(a1 + 432);
    v17 = a1 + 408;
    if ( v16 )
    {
      v18 = *(_QWORD *)(a1 + 416);
      v19 = v16 - 1;
      v20 = 1;
      v21 = v19 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v22 = v18 + 40LL * v21;
      result = 0;
      v23 = *(_QWORD *)v22;
      if ( v9 == *(_QWORD *)v22 )
      {
LABEL_6:
        v24 = *(unsigned int *)(v22 + 16);
        result = *(unsigned int *)(v22 + 20);
        v25 = v22 + 8;
        if ( v24 + 1 > result )
        {
          result = sub_C8D5F0(v22 + 8, (const void *)(v22 + 24), v24 + 1, 0x10u, v24 + 1, v19);
          v24 = *(unsigned int *)(v22 + 16);
        }
LABEL_8:
        v26 = (__int64 *)(*(_QWORD *)v25 + 16 * v24);
        *v26 = a2;
        v26[1] = a3;
        ++*(_DWORD *)(v25 + 8);
        return result;
      }
      while ( v23 != -4096 )
      {
        if ( v23 == -8192 && !result )
          result = v22;
        v21 = v19 & (v20 + v21);
        v22 = v18 + 40LL * v21;
        v23 = *(_QWORD *)v22;
        if ( v9 == *(_QWORD *)v22 )
          goto LABEL_6;
        ++v20;
      }
      v27 = *(_DWORD *)(a1 + 424);
      if ( !result )
        result = v22;
      ++*(_QWORD *)(a1 + 408);
      v28 = v27 + 1;
      if ( 4 * (v27 + 1) < 3 * v16 )
      {
        if ( v16 - *(_DWORD *)(a1 + 428) - v28 > v16 >> 3 )
        {
LABEL_20:
          *(_DWORD *)(a1 + 424) = v28;
          if ( *(_QWORD *)result != -4096 )
            --*(_DWORD *)(a1 + 428);
          *(_QWORD *)result = v9;
          v25 = result + 8;
          *(_QWORD *)(result + 8) = result + 24;
          v24 = 0;
          *(_QWORD *)(result + 16) = 0x100000000LL;
          goto LABEL_8;
        }
        sub_E8D260(v17, v16);
        v36 = *(_DWORD *)(a1 + 432);
        if ( v36 )
        {
          v37 = v36 - 1;
          v38 = 1;
          v35 = 0;
          v39 = *(_QWORD *)(a1 + 416);
          v40 = (v36 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
          v28 = *(_DWORD *)(a1 + 424) + 1;
          result = v39 + 40LL * v40;
          v41 = *(_QWORD *)result;
          if ( v9 == *(_QWORD *)result )
            goto LABEL_20;
          while ( v41 != -4096 )
          {
            if ( v41 == -8192 && !v35 )
              v35 = result;
            v40 = v37 & (v38 + v40);
            result = v39 + 40LL * v40;
            v41 = *(_QWORD *)result;
            if ( v9 == *(_QWORD *)result )
              goto LABEL_20;
            ++v38;
          }
          goto LABEL_28;
        }
        goto LABEL_44;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 408);
    }
    sub_E8D260(v17, 2 * v16);
    v29 = *(_DWORD *)(a1 + 432);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a1 + 416);
      v32 = (v29 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v28 = *(_DWORD *)(a1 + 424) + 1;
      result = v31 + 40LL * v32;
      v33 = *(_QWORD *)result;
      if ( v9 == *(_QWORD *)result )
        goto LABEL_20;
      v34 = 1;
      v35 = 0;
      while ( v33 != -4096 )
      {
        if ( !v35 && v33 == -8192 )
          v35 = result;
        v32 = v30 & (v34 + v32);
        result = v31 + 40LL * v32;
        v33 = *(_QWORD *)result;
        if ( v9 == *(_QWORD *)result )
          goto LABEL_20;
        ++v34;
      }
LABEL_28:
      if ( v35 )
        result = v35;
      goto LABEL_20;
    }
LABEL_44:
    ++*(_DWORD *)(a1 + 424);
    BUG();
  }
  v10 = *(__int64 *(__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1 + 272LL);
  if ( v10 != sub_E8DCD0 )
    return ((__int64 (__fastcall *)(__int64, __int64, __int64))v10)(a1, a2, a3);
  sub_E5CB20(*(_QWORD *)(a1 + 296), a2, (__int64)sub_E8DCD0, a4, a5, a6);
  sub_E9A490(a1, a2, a3);
  return (unsigned __int64)sub_E8DAF0(a1, a2, v11, v12, v13, v14);
}
