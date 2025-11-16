// Function: sub_1F4A4F0
// Address: 0x1f4a4f0
//
unsigned __int64 __fastcall sub_1F4A4F0(int **a1, __int64 a2)
{
  __int64 v2; // r12
  int *v3; // rbx
  int v4; // eax
  _BYTE *v5; // rax
  size_t v6; // rsi
  char *v7; // rdi
  unsigned __int64 result; // rax
  unsigned __int64 v9; // rdx
  _BYTE *v10; // rax
  __int64 v11; // rdx
  char *v12; // r14
  size_t v13; // rax
  void *v14; // rdi
  size_t v15; // r13
  __int64 v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rcx
  _BYTE *v20; // rdx
  unsigned __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rax
  size_t v24; // r14
  char *v25; // r15
  __int64 v26; // r13
  void *v27; // rdi
  __int64 v28; // rdi
  __int64 v29; // rdi
  __int64 v30; // rdi
  __int64 v31; // rdx

  v2 = a2;
  v3 = *a1;
  v4 = **a1;
  if ( v4 )
  {
    if ( v4 > 0x3FFFFFFF )
    {
      v16 = *(_QWORD *)(a2 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v16) <= 2 )
      {
        v17 = sub_16E7EE0(a2, "SS#", 3u);
      }
      else
      {
        *(_BYTE *)(v16 + 2) = 35;
        *(_WORD *)v16 = 21331;
        v17 = a2;
        *(_QWORD *)(a2 + 24) += 3LL;
      }
      sub_16E7AB0(v17, *v3 - 0x40000000);
    }
    else if ( v4 < 0 )
    {
      v19 = *((_QWORD *)v3 + 3);
      v20 = *(_BYTE **)(a2 + 24);
      v21 = *(_QWORD *)(a2 + 16);
      if ( v19
        && (v22 = v4 & 0x7FFFFFFF, (unsigned int)v22 < *(_DWORD *)(v19 + 72))
        && (v23 = *(_QWORD *)(v19 + 64) + 32 * v22, v24 = *(_QWORD *)(v23 + 8), v25 = *(char **)v23, v24) )
      {
        if ( v21 <= (unsigned __int64)v20 )
        {
          v26 = sub_16E7DE0(v2, 37);
        }
        else
        {
          v26 = v2;
          *(_QWORD *)(v2 + 24) = v20 + 1;
          *v20 = 37;
        }
        v27 = *(void **)(v26 + 24);
        if ( v24 > *(_QWORD *)(v26 + 16) - (_QWORD)v27 )
        {
          sub_16E7EE0(v26, v25, v24);
        }
        else
        {
          memcpy(v27, v25, v24);
          *(_QWORD *)(v26 + 24) += v24;
        }
      }
      else
      {
        if ( v21 <= (unsigned __int64)v20 )
        {
          v28 = sub_16E7DE0(v2, 37);
        }
        else
        {
          v28 = v2;
          *(_QWORD *)(v2 + 24) = v20 + 1;
          *v20 = 37;
        }
        sub_16E7A90(v28, *v3 & 0x7FFFFFFF);
      }
    }
    else
    {
      v5 = *(_BYTE **)(a2 + 24);
      if ( *((_QWORD *)v3 + 1) )
      {
        if ( (unsigned __int64)v5 >= *(_QWORD *)(a2 + 16) )
        {
          sub_16E7DE0(a2, 36);
        }
        else
        {
          *(_QWORD *)(a2 + 24) = v5 + 1;
          *v5 = 36;
        }
        v6 = 0;
        v7 = (char *)(*(_QWORD *)(*((_QWORD *)v3 + 1) + 72LL)
                    + *(unsigned int *)(*(_QWORD *)(*((_QWORD *)v3 + 1) + 8LL) + 24LL * (unsigned int)*v3));
        if ( v7 )
          v6 = strlen(v7);
        sub_16D1820((unsigned __int8 *)v7, v6, v2);
      }
      else
      {
        if ( (unsigned __int64)v5 >= *(_QWORD *)(a2 + 16) )
        {
          v30 = sub_16E7DE0(a2, 36);
        }
        else
        {
          v30 = a2;
          *(_QWORD *)(a2 + 24) = v5 + 1;
          *v5 = 36;
        }
        v31 = *(_QWORD *)(v30 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(v30 + 16) - v31) <= 6 )
        {
          v30 = sub_16E7EE0(v30, "physreg", 7u);
        }
        else
        {
          *(_DWORD *)v31 = 1937336432;
          *(_WORD *)(v31 + 4) = 25970;
          *(_BYTE *)(v31 + 6) = 103;
          *(_QWORD *)(v30 + 24) += 7LL;
        }
        sub_16E7A90(v30, (unsigned int)*v3);
      }
    }
  }
  else
  {
    v18 = *(_QWORD *)(a2 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v18) <= 5 )
    {
      sub_16E7EE0(a2, "$noreg", 6u);
    }
    else
    {
      *(_DWORD *)v18 = 1919905316;
      *(_WORD *)(v18 + 4) = 26469;
      *(_QWORD *)(a2 + 24) += 6LL;
    }
  }
  result = (unsigned int)v3[4];
  if ( (_DWORD)result )
  {
    v9 = *(_QWORD *)(v2 + 16);
    v10 = *(_BYTE **)(v2 + 24);
    if ( *((_QWORD *)v3 + 1) )
    {
      if ( v9 <= (unsigned __int64)v10 )
      {
        v2 = sub_16E7DE0(v2, 58);
      }
      else
      {
        *(_QWORD *)(v2 + 24) = v10 + 1;
        *v10 = 58;
      }
      result = *(_QWORD *)(*((_QWORD *)v3 + 1) + 240LL);
      v11 = (unsigned int)(v3[4] - 1);
      v12 = *(char **)(result + 8 * v11);
      if ( v12 )
      {
        v13 = strlen(*(const char **)(result + 8 * v11));
        v14 = *(void **)(v2 + 24);
        v15 = v13;
        result = *(_QWORD *)(v2 + 16) - (_QWORD)v14;
        if ( v15 > result )
        {
          return sub_16E7EE0(v2, v12, v15);
        }
        else if ( v15 )
        {
          result = (unsigned __int64)memcpy(v14, v12, v15);
          *(_QWORD *)(v2 + 24) += v15;
        }
      }
    }
    else
    {
      if ( v9 - (unsigned __int64)v10 <= 4 )
      {
        v2 = sub_16E7EE0(v2, ":sub(", 5u);
      }
      else
      {
        *(_DWORD *)v10 = 1651864378;
        v10[4] = 40;
        *(_QWORD *)(v2 + 24) += 5LL;
      }
      v29 = sub_16E7A90(v2, (unsigned int)v3[4]);
      result = *(_QWORD *)(v29 + 24);
      if ( result >= *(_QWORD *)(v29 + 16) )
      {
        return sub_16E7DE0(v29, 41);
      }
      else
      {
        *(_QWORD *)(v29 + 24) = result + 1;
        *(_BYTE *)result = 41;
      }
    }
  }
  return result;
}
