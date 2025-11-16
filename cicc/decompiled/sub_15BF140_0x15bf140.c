// Function: sub_15BF140
// Address: 0x15bf140
//
_QWORD *__fastcall sub_15BF140(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r12
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 *v8; // r14
  _QWORD *i; // rdx
  __int64 *j; // rbx
  __int64 v11; // rax
  int v12; // r15d
  __int64 v13; // rdx
  __int64 v14; // r10
  char v15; // cl
  __m128i v16; // xmm0
  int v17; // eax
  int v18; // r15d
  int v19; // eax
  __int64 v20; // rcx
  unsigned int v21; // eax
  _QWORD *v22; // rdx
  __int64 v23; // rsi
  int v24; // r8d
  _QWORD *v25; // rdi
  __int64 v26; // rdx
  _QWORD *k; // rdx
  __int64 v28; // [rsp+8h] [rbp-98h]
  int v29; // [rsp+1Ch] [rbp-84h] BYREF
  __int64 v30; // [rsp+20h] [rbp-80h] BYREF
  __int64 v31; // [rsp+28h] [rbp-78h] BYREF
  __int64 v32; // [rsp+30h] [rbp-70h] BYREF
  __int64 v33; // [rsp+38h] [rbp-68h] BYREF
  __m128i v34; // [rsp+40h] [rbp-60h]
  __int64 v36; // [rsp+58h] [rbp-48h]
  char v37; // [rsp+60h] [rbp-40h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(__int64 **)(a1 + 8);
  v5 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16)
      | (((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
      | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
      | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
      | (unsigned int)(a2 - 1)
      | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1))
     + 1;
  if ( (unsigned int)v5 < 0x40 )
    LODWORD(v5) = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_22077B0(8LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[v3];
    for ( i = &result[v7]; i != result; ++result )
    {
      if ( result )
        *result = -8;
    }
    for ( j = v4; v8 != j; ++j )
    {
      v11 = *j;
      if ( *j != -16 && v11 != -8 )
      {
        v12 = *(_DWORD *)(a1 + 24);
        if ( !v12 )
        {
          MEMORY[0] = *j;
          BUG();
        }
        v13 = *(unsigned int *)(v11 + 8);
        v14 = *(_QWORD *)(a1 + 8);
        v32 = *(_QWORD *)(v11 - 8 * v13);
        v15 = *(_BYTE *)(v11 + 56);
        v33 = *(_QWORD *)(v11 + 8 * (1 - v13));
        if ( *(_BYTE *)(v11 + 40) )
        {
          v16 = _mm_loadu_si128((const __m128i *)(v11 + 24));
          v37 = v15;
          v34 = v16;
          if ( v15 )
          {
            v36 = *(_QWORD *)(v11 + 48);
            v31 = v36;
          }
          else
          {
            v31 = 0;
          }
          v30 = v34.m128i_i64[1];
          v17 = v34.m128i_i32[0];
        }
        else
        {
          v37 = v15;
          if ( v15 )
          {
            v36 = *(_QWORD *)(v11 + 48);
            v31 = v36;
          }
          else
          {
            v31 = 0;
          }
          v30 = 0;
          v17 = 0;
        }
        v28 = v14;
        v18 = v12 - 1;
        v29 = v17;
        v19 = sub_15B5960(&v32, &v33, &v29, &v30, &v31);
        v20 = *j;
        v21 = v18 & v19;
        v22 = (_QWORD *)(v28 + 8LL * v21);
        v23 = *v22;
        if ( *v22 != *j )
        {
          v24 = 1;
          v25 = 0;
          while ( v23 != -8 )
          {
            if ( v25 || v23 != -16 )
              v22 = v25;
            v21 = v18 & (v24 + v21);
            v23 = *(_QWORD *)(v28 + 8LL * v21);
            if ( v23 == v20 )
            {
              v22 = (_QWORD *)(v28 + 8LL * v21);
              goto LABEL_25;
            }
            ++v24;
            v25 = v22;
            v22 = (_QWORD *)(v28 + 8LL * v21);
          }
          if ( v25 )
            v22 = v25;
        }
LABEL_25:
        *v22 = v20;
        ++*(_DWORD *)(a1 + 16);
      }
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v26 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[v26]; k != result; ++result )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
