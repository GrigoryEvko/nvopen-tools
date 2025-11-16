// Function: sub_1A2CCD0
// Address: 0x1a2ccd0
//
__int64 __fastcall sub_1A2CCD0(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  char v4; // r12
  unsigned int v5; // eax
  int v6; // r14d
  __int64 v7; // r15
  _QWORD *v8; // rax
  _BYTE *v9; // r12
  _QWORD *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx
  _QWORD *v13; // rdx
  _QWORD *v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // r8
  int v17; // edi
  int v18; // r14d
  _QWORD *v19; // r10
  unsigned int v20; // esi
  _QWORD *v21; // rdx
  __int64 v22; // r9
  __int64 *v23; // r12
  __int64 v24; // r13
  bool v25; // zf
  __int64 *v26; // rsi
  _QWORD *v27; // rax
  __int64 v28; // rdx
  _QWORD *i; // rdx
  __int64 *j; // rax
  __int64 v31; // rdx
  __int64 v32; // r10
  int v33; // r9d
  int v34; // r14d
  _QWORD *v35; // r13
  unsigned int v36; // edi
  _QWORD *v37; // rcx
  __int64 v38; // r11
  __int64 v39; // rdx
  int v40; // ecx
  __int64 v41; // rax
  int v42; // edx
  _BYTE v43[112]; // [rsp+10h] [rbp-70h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 3 )
  {
    if ( v4 )
      return result;
    v23 = *(__int64 **)(a1 + 16);
    v24 = *(unsigned int *)(a1 + 24);
    *(_BYTE *)(a1 + 8) = result | 1;
  }
  else
  {
    v5 = sub_1454B60(a2 - 1);
    v6 = v5;
    if ( v5 > 0x40 )
    {
      v7 = 2LL * v5;
      if ( v4 )
        goto LABEL_5;
      v23 = *(__int64 **)(a1 + 16);
      v24 = *(unsigned int *)(a1 + 24);
    }
    else
    {
      if ( v4 )
      {
        v7 = 128;
        v6 = 64;
LABEL_5:
        v8 = (_QWORD *)(a1 + 16);
        v9 = v43;
        do
        {
          if ( *v8 <= 0xFFFFFFFFFFFFFFFDLL )
          {
            if ( v9 )
              *(_QWORD *)v9 = *v8;
            v9 += 16;
            *((_QWORD *)v9 - 1) = v8[1];
          }
          v8 += 2;
        }
        while ( v8 != (_QWORD *)(a1 + 80) );
        *(_BYTE *)(a1 + 8) &= ~1u;
        v10 = (_QWORD *)sub_22077B0(v7 * 8);
        v11 = *(_QWORD *)(a1 + 8);
        *(_DWORD *)(a1 + 24) = v6;
        *(_QWORD *)(a1 + 16) = v10;
        v12 = v11 & 1;
        *(_QWORD *)(a1 + 8) = v12;
        if ( (_BYTE)v12 )
        {
          v10 = (_QWORD *)(a1 + 16);
          v7 = 8;
        }
        v13 = v10;
        v14 = &v10[v7];
        while ( 1 )
        {
          if ( v13 )
            *v10 = -1;
          v10 += 2;
          if ( v14 == v10 )
            break;
          v13 = v10;
        }
        result = (__int64)v43;
        if ( v9 != v43 )
        {
          while ( 1 )
          {
            v15 = *(_QWORD *)result;
            if ( *(_QWORD *)result <= 0xFFFFFFFFFFFFFFFDLL )
            {
              if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
              {
                v16 = a1 + 16;
                v17 = 3;
              }
              else
              {
                v42 = *(_DWORD *)(a1 + 24);
                v16 = *(_QWORD *)(a1 + 16);
                if ( !v42 )
                  goto LABEL_71;
                v17 = v42 - 1;
              }
              v18 = 1;
              v19 = 0;
              v20 = v17 & (37 * v15);
              v21 = (_QWORD *)(v16 + 16LL * v20);
              v22 = *v21;
              if ( v15 != *v21 )
              {
                while ( v22 != -1 )
                {
                  if ( v22 == -2 && !v19 )
                    v19 = v21;
                  v20 = v17 & (v18 + v20);
                  v21 = (_QWORD *)(v16 + 16LL * v20);
                  v22 = *v21;
                  if ( v15 == *v21 )
                    goto LABEL_22;
                  ++v18;
                }
                if ( v19 )
                  v21 = v19;
              }
LABEL_22:
              *v21 = v15;
              v21[1] = *(_QWORD *)(result + 8);
              *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
            }
            result += 16;
            if ( v9 == (_BYTE *)result )
              return result;
          }
        }
        return result;
      }
      v23 = *(__int64 **)(a1 + 16);
      v24 = *(unsigned int *)(a1 + 24);
      v7 = 128;
      v6 = 64;
    }
    v41 = sub_22077B0(v7 * 8);
    *(_DWORD *)(a1 + 24) = v6;
    *(_QWORD *)(a1 + 16) = v41;
  }
  v25 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v26 = &v23[2 * v24];
  if ( v25 )
  {
    v27 = *(_QWORD **)(a1 + 16);
    v28 = 2LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v27 = (_QWORD *)(a1 + 16);
    v28 = 8;
  }
  for ( i = &v27[v28]; i != v27; v27 += 2 )
  {
    if ( v27 )
      *v27 = -1;
  }
  for ( j = v23; v26 != j; *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1 )
  {
    while ( 1 )
    {
      v31 = *j;
      if ( (unsigned __int64)*j <= 0xFFFFFFFFFFFFFFFDLL )
        break;
      j += 2;
      if ( v26 == j )
        return j___libc_free_0(v23);
    }
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v32 = a1 + 16;
      v33 = 3;
    }
    else
    {
      v40 = *(_DWORD *)(a1 + 24);
      v32 = *(_QWORD *)(a1 + 16);
      if ( !v40 )
      {
LABEL_71:
        MEMORY[0] = 0;
        BUG();
      }
      v33 = v40 - 1;
    }
    v34 = 1;
    v35 = 0;
    v36 = v33 & (37 * v31);
    v37 = (_QWORD *)(v32 + 16LL * v36);
    v38 = *v37;
    if ( v31 != *v37 )
    {
      while ( v38 != -1 )
      {
        if ( v38 == -2 && !v35 )
          v35 = v37;
        v36 = v33 & (v34 + v36);
        v37 = (_QWORD *)(v32 + 16LL * v36);
        v38 = *v37;
        if ( v31 == *v37 )
          goto LABEL_39;
        ++v34;
      }
      if ( v35 )
        v37 = v35;
    }
LABEL_39:
    *v37 = v31;
    v39 = j[1];
    j += 2;
    v37[1] = v39;
  }
  return j___libc_free_0(v23);
}
