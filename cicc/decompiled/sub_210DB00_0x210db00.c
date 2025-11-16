// Function: sub_210DB00
// Address: 0x210db00
//
__int64 __fastcall sub_210DB00(__int64 a1, __int64 a2, const void *a3, __int64 a4)
{
  size_t v4; // r15
  char *v7; // r13
  char *v8; // r14
  unsigned int v9; // esi
  __int64 v10; // r8
  unsigned int v11; // edi
  __int64 result; // rax
  __int64 v13; // rcx
  __int64 v14; // rdi
  __int64 v15; // rsi
  int v16; // r11d
  _QWORD *v17; // rdx
  int v18; // eax
  int v19; // ecx
  int v20; // eax
  int v21; // esi
  __int64 v22; // rdi
  __int64 v23; // r8
  int v24; // r10d
  _QWORD *v25; // r9
  int v26; // eax
  __int64 v27; // rdi
  _QWORD *v28; // r8
  unsigned int v29; // r15d
  int v30; // r9d
  __int64 v31; // rsi
  int v32; // edx
  __int64 *v33; // r11

  v4 = 4 * a4;
  if ( (unsigned __int64)(4 * a4) > 0x7FFFFFFFFFFFFFFCLL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  if ( v4 )
  {
    v7 = (char *)sub_22077B0(4 * a4);
    v8 = &v7[v4];
    memcpy(v7, a3, v4);
  }
  else
  {
    v8 = 0;
    v7 = 0;
  }
  v9 = *(_DWORD *)(a1 + 184);
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 160);
    goto LABEL_21;
  }
  v10 = *(_QWORD *)(a1 + 168);
  v11 = (v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = v10 + 32LL * v11;
  v13 = *(_QWORD *)result;
  if ( a2 != *(_QWORD *)result )
  {
    v16 = 1;
    v17 = 0;
    while ( v13 != -8 )
    {
      if ( v13 != -16 || v17 )
        result = (__int64)v17;
      v32 = v16 + 1;
      v11 = (v9 - 1) & (v16 + v11);
      v33 = (__int64 *)(v10 + 32LL * v11);
      v13 = *v33;
      if ( a2 == *v33 )
      {
        v14 = v33[1];
        result = (__int64)v33;
        v15 = v33[3] - v14;
        goto LABEL_7;
      }
      v16 = v32;
      v17 = (_QWORD *)result;
      result = v10 + 32LL * v11;
    }
    if ( !v17 )
      v17 = (_QWORD *)result;
    v18 = *(_DWORD *)(a1 + 176);
    ++*(_QWORD *)(a1 + 160);
    v19 = v18 + 1;
    if ( 4 * (v18 + 1) < 3 * v9 )
    {
      result = v9 - *(_DWORD *)(a1 + 180) - v19;
      if ( (unsigned int)result > v9 >> 3 )
      {
LABEL_15:
        *(_DWORD *)(a1 + 176) = v19;
        if ( *v17 != -8 )
          --*(_DWORD *)(a1 + 180);
        *v17 = a2;
        v17[1] = v7;
        v17[2] = v8;
        v17[3] = v8;
        return result;
      }
      sub_210D890(a1 + 160, v9);
      v26 = *(_DWORD *)(a1 + 184);
      if ( v26 )
      {
        result = (unsigned int)(v26 - 1);
        v27 = *(_QWORD *)(a1 + 168);
        v28 = 0;
        v29 = result & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v30 = 1;
        v19 = *(_DWORD *)(a1 + 176) + 1;
        v17 = (_QWORD *)(v27 + 32LL * v29);
        v31 = *v17;
        if ( a2 != *v17 )
        {
          while ( v31 != -8 )
          {
            if ( !v28 && v31 == -16 )
              v28 = v17;
            v29 = result & (v30 + v29);
            v17 = (_QWORD *)(v27 + 32LL * v29);
            v31 = *v17;
            if ( a2 == *v17 )
              goto LABEL_15;
            ++v30;
          }
          if ( v28 )
            v17 = v28;
        }
        goto LABEL_15;
      }
LABEL_51:
      ++*(_DWORD *)(a1 + 176);
      BUG();
    }
LABEL_21:
    sub_210D890(a1 + 160, 2 * v9);
    v20 = *(_DWORD *)(a1 + 184);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a1 + 168);
      result = (v20 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v19 = *(_DWORD *)(a1 + 176) + 1;
      v17 = (_QWORD *)(v22 + 32 * result);
      v23 = *v17;
      if ( a2 != *v17 )
      {
        v24 = 1;
        v25 = 0;
        while ( v23 != -8 )
        {
          if ( !v25 && v23 == -16 )
            v25 = v17;
          result = v21 & (unsigned int)(v24 + result);
          v17 = (_QWORD *)(v22 + 32LL * (unsigned int)result);
          v23 = *v17;
          if ( a2 == *v17 )
            goto LABEL_15;
          ++v24;
        }
        if ( v25 )
          v17 = v25;
      }
      goto LABEL_15;
    }
    goto LABEL_51;
  }
  v14 = *(_QWORD *)(result + 8);
  v15 = *(_QWORD *)(result + 24) - v14;
LABEL_7:
  *(_QWORD *)(result + 8) = v7;
  *(_QWORD *)(result + 16) = v8;
  *(_QWORD *)(result + 24) = v8;
  if ( v14 )
    return j_j___libc_free_0(v14, v15);
  return result;
}
