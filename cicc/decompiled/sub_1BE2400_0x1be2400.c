// Function: sub_1BE2400
// Address: 0x1be2400
//
__int64 __fastcall sub_1BE2400(__int64 a1, __int64 a2)
{
  unsigned __int16 v3; // r12
  __int64 v4; // rdx
  __int64 v5; // rdi
  __int64 v6; // rdx
  int v7; // edi
  char *v8; // rax
  char *v9; // r12
  size_t v10; // rax
  void *v11; // rdi
  size_t v12; // r14
  __int64 *v13; // r12
  __int64 result; // rax
  __int64 *i; // r14
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // r13
  _BYTE *v20; // rax
  __int64 v21; // rdx

  v3 = a1 + 40;
  v4 = *(_QWORD *)(a2 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v4) <= 2 )
  {
    v5 = sub_16E7EE0(a2, "%vp", 3u);
  }
  else
  {
    *(_BYTE *)(v4 + 2) = 112;
    *(_WORD *)v4 = 30245;
    v5 = a2;
    *(_QWORD *)(a2 + 24) += 3LL;
  }
  sub_16E7AB0(v5, v3);
  v6 = *(_QWORD *)(a2 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v6) <= 2 )
  {
    sub_16E7EE0(a2, " = ", 3u);
  }
  else
  {
    *(_BYTE *)(v6 + 2) = 32;
    *(_WORD *)v6 = 15648;
    *(_QWORD *)(a2 + 24) += 3LL;
  }
  v7 = *(unsigned __int8 *)(a1 + 112);
  if ( v7 == 66 )
  {
    v21 = *(_QWORD *)(a2 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v21) <= 2 )
    {
      sub_16E7EE0(a2, "not", 3u);
    }
    else
    {
      *(_BYTE *)(v21 + 2) = 116;
      *(_WORD *)v21 = 28526;
      *(_QWORD *)(a2 + 24) += 3LL;
    }
  }
  else
  {
    v8 = sub_15F29F0(v7);
    v9 = v8;
    if ( v8 )
    {
      v10 = strlen(v8);
      v11 = *(void **)(a2 + 24);
      v12 = v10;
      if ( v10 > *(_QWORD *)(a2 + 16) - (_QWORD)v11 )
      {
        sub_16E7EE0(a2, v9, v10);
      }
      else if ( v10 )
      {
        memcpy(v11, v9, v10);
        *(_QWORD *)(a2 + 24) += v12;
      }
    }
  }
  v13 = *(__int64 **)(a1 + 80);
  result = *(unsigned int *)(a1 + 88);
  for ( i = &v13[result]; i != v13; result = sub_16E7AB0(v18, (unsigned __int16)v19) )
  {
    v19 = *v13;
    v20 = *(_BYTE **)(a2 + 24);
    if ( *(_BYTE **)(a2 + 16) == v20 )
    {
      sub_16E7EE0(a2, " ", 1u);
      v16 = *(_QWORD *)(a2 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v16) <= 2 )
      {
LABEL_17:
        v18 = sub_16E7EE0(a2, "%vp", 3u);
        goto LABEL_14;
      }
    }
    else
    {
      *v20 = 32;
      v16 = *(_QWORD *)(a2 + 24) + 1LL;
      v17 = *(_QWORD *)(a2 + 16);
      *(_QWORD *)(a2 + 24) = v16;
      if ( (unsigned __int64)(v17 - v16) <= 2 )
        goto LABEL_17;
    }
    *(_BYTE *)(v16 + 2) = 112;
    v18 = a2;
    *(_WORD *)v16 = 30245;
    *(_QWORD *)(a2 + 24) += 3LL;
LABEL_14:
    ++v13;
  }
  return result;
}
