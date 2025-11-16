// Function: sub_BD6500
// Address: 0xbd6500
//
__int64 __fastcall sub_BD6500(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v5; // r13
  unsigned int v6; // esi
  __int64 v7; // r8
  __int64 v8; // rdi
  int v9; // r15d
  __int64 *v10; // r10
  unsigned int v11; // ecx
  __int64 *v12; // rax
  __int64 v13; // rdx
  int v14; // edx
  __int64 v15; // rdi
  int v16; // esi
  unsigned int v17; // ecx
  __int64 *v18; // rdx
  __int64 v19; // r8
  int v20; // eax
  int v21; // esi
  __int64 v22; // rdi
  unsigned int v23; // ecx
  int v24; // edx
  __int64 v25; // rax
  int v26; // eax
  int v27; // eax
  int v28; // ecx
  __int64 v29; // rsi
  int v30; // r8d
  __int64 *v31; // rdi
  unsigned int v32; // r14d
  __int64 v33; // rax
  int v34; // edx
  int v35; // r9d
  int v36; // r9d
  __int64 *v37; // r8

  result = sub_BD5C60(a1);
  if ( a2 )
  {
    *(_BYTE *)(a1 + 7) |= 0x10u;
    v5 = *(_QWORD *)result;
    v6 = *(_DWORD *)(*(_QWORD *)result + 200LL);
    v7 = *(_QWORD *)result + 176LL;
    if ( v6 )
    {
      v8 = *(_QWORD *)(v5 + 184);
      v9 = 1;
      v10 = 0;
      v11 = (v6 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v12 = (__int64 *)(v8 + 16LL * v11);
      v13 = *v12;
      if ( a1 == *v12 )
      {
LABEL_4:
        result = (__int64)(v12 + 1);
LABEL_5:
        *(_QWORD *)result = a2;
        return result;
      }
      while ( v13 != -4096 )
      {
        if ( !v10 && v13 == -8192 )
          v10 = v12;
        v11 = (v6 - 1) & (v9 + v11);
        v12 = (__int64 *)(v8 + 16LL * v11);
        v13 = *v12;
        if ( a1 == *v12 )
          goto LABEL_4;
        ++v9;
      }
      if ( !v10 )
        v10 = v12;
      v26 = *(_DWORD *)(v5 + 192);
      ++*(_QWORD *)(v5 + 176);
      v24 = v26 + 1;
      if ( 4 * (v26 + 1) < 3 * v6 )
      {
        if ( v6 - *(_DWORD *)(v5 + 196) - v24 > v6 >> 3 )
        {
LABEL_14:
          *(_DWORD *)(v5 + 192) = v24;
          if ( *v10 != -4096 )
            --*(_DWORD *)(v5 + 196);
          *v10 = a1;
          result = (__int64)(v10 + 1);
          v10[1] = 0;
          goto LABEL_5;
        }
        sub_BD6320(v7, v6);
        v27 = *(_DWORD *)(v5 + 200);
        if ( v27 )
        {
          v28 = v27 - 1;
          v29 = *(_QWORD *)(v5 + 184);
          v30 = 1;
          v31 = 0;
          v32 = (v27 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
          v24 = *(_DWORD *)(v5 + 192) + 1;
          v10 = (__int64 *)(v29 + 16LL * v32);
          v33 = *v10;
          if ( a1 != *v10 )
          {
            while ( v33 != -4096 )
            {
              if ( v33 == -8192 && !v31 )
                v31 = v10;
              v32 = v28 & (v30 + v32);
              v10 = (__int64 *)(v29 + 16LL * v32);
              v33 = *v10;
              if ( a1 == *v10 )
                goto LABEL_14;
              ++v30;
            }
            if ( v31 )
              v10 = v31;
          }
          goto LABEL_14;
        }
LABEL_52:
        ++*(_DWORD *)(v5 + 192);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(v5 + 176);
    }
    sub_BD6320(v7, 2 * v6);
    v20 = *(_DWORD *)(v5 + 200);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(v5 + 184);
      v23 = (v20 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v24 = *(_DWORD *)(v5 + 192) + 1;
      v10 = (__int64 *)(v22 + 16LL * v23);
      v25 = *v10;
      if ( a1 != *v10 )
      {
        v36 = 1;
        v37 = 0;
        while ( v25 != -4096 )
        {
          if ( !v37 && v25 == -8192 )
            v37 = v10;
          v23 = v21 & (v36 + v23);
          v10 = (__int64 *)(v22 + 16LL * v23);
          v25 = *v10;
          if ( a1 == *v10 )
            goto LABEL_14;
          ++v36;
        }
        if ( v37 )
          v10 = v37;
      }
      goto LABEL_14;
    }
    goto LABEL_52;
  }
  if ( (*(_BYTE *)(a1 + 7) & 0x10) != 0 )
  {
    result = *(_QWORD *)result;
    v14 = *(_DWORD *)(result + 200);
    v15 = *(_QWORD *)(result + 184);
    if ( v14 )
    {
      v16 = v14 - 1;
      v17 = (v14 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v18 = (__int64 *)(v15 + 16LL * v17);
      v19 = *v18;
      if ( a1 == *v18 )
      {
LABEL_10:
        *v18 = -8192;
        --*(_DWORD *)(result + 192);
        ++*(_DWORD *)(result + 196);
      }
      else
      {
        v34 = 1;
        while ( v19 != -4096 )
        {
          v35 = v34 + 1;
          v17 = v16 & (v34 + v17);
          v18 = (__int64 *)(v15 + 16LL * v17);
          v19 = *v18;
          if ( a1 == *v18 )
            goto LABEL_10;
          v34 = v35;
        }
      }
    }
  }
  *(_BYTE *)(a1 + 7) &= ~0x10u;
  return result;
}
