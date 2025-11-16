// Function: sub_2FF5BC0
// Address: 0x2ff5bc0
//
unsigned __int64 __fastcall sub_2FF5BC0(int **a1, __int64 a2)
{
  __int64 v2; // r12
  int *v3; // rbx
  int v4; // eax
  __int64 v5; // rdx
  _BYTE *v6; // rax
  size_t v7; // rsi
  char *v8; // rdi
  unsigned __int64 result; // rax
  unsigned __int64 v10; // rdx
  _BYTE *v11; // rax
  __int64 v12; // rdx
  unsigned __int8 *v13; // r14
  size_t v14; // rax
  void *v15; // rdi
  size_t v16; // r13
  __int64 v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // rcx
  _BYTE *v21; // rdx
  unsigned __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rax
  size_t v25; // r14
  void *v26; // r15
  __int64 v27; // r13
  void *v28; // rdi
  __int64 v29; // rdi
  __int64 v30; // rdi
  _BYTE *v31; // rax
  __int64 v32; // rdi
  __int64 v33; // rdx

  v2 = a2;
  v3 = *a1;
  v4 = **a1;
  if ( v4 )
  {
    if ( (unsigned int)(v4 - 0x40000000) <= 0x3FFFFFFF )
    {
      v17 = *(_QWORD *)(a2 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v17) <= 2 )
      {
        v18 = sub_CB6200(a2, "SS#", 3u);
      }
      else
      {
        *(_BYTE *)(v17 + 2) = 35;
        *(_WORD *)v17 = 21331;
        v18 = a2;
        *(_QWORD *)(a2 + 32) += 3LL;
      }
      sub_CB59F0(v18, *v3 - 0x40000000);
    }
    else if ( v4 < 0 )
    {
      v20 = *((_QWORD *)v3 + 3);
      v21 = *(_BYTE **)(a2 + 32);
      v22 = *(_QWORD *)(a2 + 24);
      if ( v20
        && (v23 = v4 & 0x7FFFFFFF, (unsigned int)v23 < *(_DWORD *)(v20 + 104))
        && (v24 = *(_QWORD *)(v20 + 96) + 32 * v23, v25 = *(_QWORD *)(v24 + 8), v26 = *(void **)v24, v25) )
      {
        if ( v22 <= (unsigned __int64)v21 )
        {
          v27 = sub_CB5D20(v2, 37);
        }
        else
        {
          v27 = v2;
          *(_QWORD *)(v2 + 32) = v21 + 1;
          *v21 = 37;
        }
        v28 = *(void **)(v27 + 32);
        if ( v25 <= *(_QWORD *)(v27 + 24) - (_QWORD)v28 )
        {
          memcpy(v28, v26, v25);
          *(_QWORD *)(v27 + 32) += v25;
        }
        else
        {
          sub_CB6200(v27, (unsigned __int8 *)v26, v25);
        }
      }
      else
      {
        if ( v22 <= (unsigned __int64)v21 )
        {
          v29 = sub_CB5D20(v2, 37);
        }
        else
        {
          v29 = v2;
          *(_QWORD *)(v2 + 32) = v21 + 1;
          *v21 = 37;
        }
        sub_CB59D0(v29, *v3 & 0x7FFFFFFF);
      }
    }
    else
    {
      v5 = *((_QWORD *)v3 + 1);
      if ( v5 )
      {
        if ( (unsigned int)v4 >= *(_DWORD *)(v5 + 16) )
          BUG();
        v6 = *(_BYTE **)(a2 + 32);
        if ( (unsigned __int64)v6 >= *(_QWORD *)(a2 + 24) )
        {
          sub_CB5D20(a2, 36);
        }
        else
        {
          *(_QWORD *)(a2 + 32) = v6 + 1;
          *v6 = 36;
        }
        v7 = 0;
        v8 = (char *)(*(_QWORD *)(*((_QWORD *)v3 + 1) + 72LL)
                    + *(unsigned int *)(*(_QWORD *)(*((_QWORD *)v3 + 1) + 8LL) + 24LL * (unsigned int)*v3));
        if ( v8 )
          v7 = strlen(v8);
        sub_C925A0((unsigned __int8 *)v8, v7, v2);
      }
      else
      {
        v31 = *(_BYTE **)(a2 + 32);
        if ( (unsigned __int64)v31 >= *(_QWORD *)(a2 + 24) )
        {
          v32 = sub_CB5D20(a2, 36);
        }
        else
        {
          v32 = a2;
          *(_QWORD *)(a2 + 32) = v31 + 1;
          *v31 = 36;
        }
        v33 = *(_QWORD *)(v32 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(v32 + 24) - v33) <= 6 )
        {
          v32 = sub_CB6200(v32, "physreg", 7u);
        }
        else
        {
          *(_DWORD *)v33 = 1937336432;
          *(_WORD *)(v33 + 4) = 25970;
          *(_BYTE *)(v33 + 6) = 103;
          *(_QWORD *)(v32 + 32) += 7LL;
        }
        sub_CB59D0(v32, (unsigned int)*v3);
      }
    }
  }
  else
  {
    v19 = *(_QWORD *)(a2 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v19) <= 5 )
    {
      sub_CB6200(a2, "$noreg", 6u);
    }
    else
    {
      *(_DWORD *)v19 = 1919905316;
      *(_WORD *)(v19 + 4) = 26469;
      *(_QWORD *)(a2 + 32) += 6LL;
    }
  }
  result = (unsigned int)v3[4];
  if ( (_DWORD)result )
  {
    v10 = *(_QWORD *)(v2 + 24);
    v11 = *(_BYTE **)(v2 + 32);
    if ( *((_QWORD *)v3 + 1) )
    {
      if ( (unsigned __int64)v11 >= v10 )
      {
        v2 = sub_CB5D20(v2, 58);
      }
      else
      {
        *(_QWORD *)(v2 + 32) = v11 + 1;
        *v11 = 58;
      }
      result = *(_QWORD *)(*((_QWORD *)v3 + 1) + 256LL);
      v12 = (unsigned int)(v3[4] - 1);
      v13 = *(unsigned __int8 **)(result + 8 * v12);
      if ( v13 )
      {
        v14 = strlen(*(const char **)(result + 8 * v12));
        v15 = *(void **)(v2 + 32);
        v16 = v14;
        result = *(_QWORD *)(v2 + 24) - (_QWORD)v15;
        if ( v16 > result )
        {
          return sub_CB6200(v2, v13, v16);
        }
        else if ( v16 )
        {
          result = (unsigned __int64)memcpy(v15, v13, v16);
          *(_QWORD *)(v2 + 32) += v16;
        }
      }
    }
    else
    {
      if ( v10 - (unsigned __int64)v11 <= 4 )
      {
        v2 = sub_CB6200(v2, ":sub(", 5u);
      }
      else
      {
        *(_DWORD *)v11 = 1651864378;
        v11[4] = 40;
        *(_QWORD *)(v2 + 32) += 5LL;
      }
      v30 = sub_CB59D0(v2, (unsigned int)v3[4]);
      result = *(_QWORD *)(v30 + 32);
      if ( result >= *(_QWORD *)(v30 + 24) )
      {
        return sub_CB5D20(v30, 41);
      }
      else
      {
        *(_QWORD *)(v30 + 32) = result + 1;
        *(_BYTE *)result = 41;
      }
    }
  }
  return result;
}
