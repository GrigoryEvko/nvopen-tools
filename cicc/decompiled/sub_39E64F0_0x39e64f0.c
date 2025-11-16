// Function: sub_39E64F0
// Address: 0x39e64f0
//
_BYTE *__fastcall sub_39E64F0(__int64 a1, unsigned int a2, __int64 a3, int a4, unsigned int a5)
{
  char v7; // r13
  unsigned int v9; // ebx
  _BYTE *result; // rax
  __int64 v11; // rdi
  _BYTE *v12; // rax
  __int64 v13; // rdi
  _WORD *v14; // rdx
  unsigned __int64 v15; // r13
  __int64 v16; // rdi
  _QWORD *v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // r14
  char *v20; // rsi
  size_t v21; // rdx
  void *v22; // rdi
  __int64 v23; // rdi
  _WORD *v24; // rdx
  __int64 v25; // rax

  v7 = a4;
  if ( !a2 || (a2 & (a2 - 1)) != 0 )
  {
    if ( a4 == 2 )
    {
      v16 = *(_QWORD *)(a1 + 272);
      v17 = *(_QWORD **)(v16 + 24);
      if ( *(_QWORD *)(v16 + 16) - (_QWORD)v17 <= 7u )
      {
        sub_16E7EE0(v16, ".balignw", 8u);
      }
      else
      {
        *v17 = 0x776E67696C61622ELL;
        *(_QWORD *)(v16 + 24) += 8LL;
      }
    }
    else if ( a4 == 4 )
    {
      sub_1263B40(*(_QWORD *)(a1 + 272), ".balignl");
    }
    else
    {
      sub_1263B40(*(_QWORD *)(a1 + 272), ".balign");
    }
    v11 = *(_QWORD *)(a1 + 272);
    v12 = *(_BYTE **)(v11 + 24);
    if ( (unsigned __int64)v12 >= *(_QWORD *)(v11 + 16) )
    {
      v11 = sub_16E7DE0(v11, 32);
    }
    else
    {
      *(_QWORD *)(v11 + 24) = v12 + 1;
      *v12 = 32;
    }
    sub_16E7A90(v11, a2);
    v13 = *(_QWORD *)(a1 + 272);
    v14 = *(_WORD **)(v13 + 24);
    if ( *(_QWORD *)(v13 + 16) - (_QWORD)v14 <= 1u )
    {
      v13 = sub_16E7EE0(v13, ", ", 2u);
    }
    else
    {
      *v14 = 8236;
      *(_QWORD *)(v13 + 24) += 2LL;
    }
    sub_16E7AB0(v13, a3 & (0xFFFFFFFFFFFFFFFFLL >> (8 * (8 - v7))));
    if ( a5 )
    {
      v23 = *(_QWORD *)(a1 + 272);
      v24 = *(_WORD **)(v23 + 24);
      if ( *(_QWORD *)(v23 + 16) - (_QWORD)v24 <= 1u )
      {
        v23 = sub_16E7EE0(v23, ", ", 2u);
      }
      else
      {
        *v24 = 8236;
        *(_QWORD *)(v23 + 24) += 2LL;
      }
      sub_16E7A90(v23, a5);
    }
    v15 = *(unsigned int *)(a1 + 312);
    if ( *(_DWORD *)(a1 + 312) )
    {
      v19 = *(_QWORD *)(a1 + 272);
      v20 = *(char **)(a1 + 304);
      v21 = *(unsigned int *)(a1 + 312);
      v22 = *(void **)(v19 + 24);
      if ( v15 > *(_QWORD *)(v19 + 16) - (_QWORD)v22 )
      {
        sub_16E7EE0(*(_QWORD *)(a1 + 272), v20, v21);
      }
      else
      {
        memcpy(v22, v20, v21);
        *(_QWORD *)(v19 + 24) += v15;
      }
    }
    *(_DWORD *)(a1 + 312) = 0;
    if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    {
      return sub_39E0440(a1);
    }
    else
    {
      v18 = *(_QWORD *)(a1 + 272);
      result = *(_BYTE **)(v18 + 24);
      if ( (unsigned __int64)result >= *(_QWORD *)(v18 + 16) )
      {
        return (_BYTE *)sub_16E7DE0(v18, 10);
      }
      else
      {
        *(_QWORD *)(v18 + 24) = result + 1;
        *result = 10;
      }
    }
  }
  else
  {
    if ( a4 == 2 )
    {
      sub_1263B40(*(_QWORD *)(a1 + 272), ".p2alignw ");
    }
    else if ( a4 == 4 )
    {
      sub_1263B40(*(_QWORD *)(a1 + 272), ".p2alignl ");
    }
    else
    {
      sub_1263B40(*(_QWORD *)(a1 + 272), "\t.p2align\t");
    }
    _BitScanReverse(&v9, a2);
    sub_16E7A90(*(_QWORD *)(a1 + 272), 31 - (v9 ^ 0x1F));
    if ( a3 || a5 )
    {
      sub_1263B40(*(_QWORD *)(a1 + 272), ", 0x");
      sub_16E7B10(*(_QWORD *)(a1 + 272), a3 & (0xFFFFFFFFFFFFFFFFLL >> (8 * (8 - v7))));
      if ( a5 )
      {
        v25 = sub_1263B40(*(_QWORD *)(a1 + 272), ", ");
        sub_16E7A90(v25, a5);
      }
    }
    return sub_39E06C0(a1);
  }
  return result;
}
