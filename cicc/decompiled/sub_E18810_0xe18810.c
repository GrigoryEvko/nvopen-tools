// Function: sub_E18810
// Address: 0xe18810
//
__int64 __fastcall sub_E18810(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r9
  __int64 v6; // rax
  __int64 v7; // r12
  _BYTE *v9; // rdx
  _BYTE *v11; // rax
  __int64 v12; // rcx
  unsigned __int64 v13; // rax
  _QWORD *v14; // rax
  _BYTE *v15; // rdx
  _BYTE *v16; // rax
  unsigned __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r8
  __int64 v21; // r9
  char v22; // al
  __int64 *v23; // r13
  __int64 v24; // rcx
  size_t v25; // r15
  __int64 v26; // r14
  __int64 v27; // rax
  __int64 *v28; // r15
  void *v29; // rdi
  __int64 *src; // [rsp+8h] [rbp-58h]
  unsigned __int64 v31; // [rsp+18h] [rbp-48h] BYREF
  unsigned __int64 v32; // [rsp+20h] [rbp-40h] BYREF
  __int64 v33[7]; // [rsp+28h] [rbp-38h] BYREF

  v5 = *(_QWORD *)a1;
  v6 = *(_QWORD *)(a1 + 8);
  if ( *(_QWORD *)a1 == v6 || *(_BYTE *)v5 != 84 )
    return 0;
  v9 = (_BYTE *)(v5 + 1);
  v31 = 0;
  *(_QWORD *)a1 = v5 + 1;
  if ( v6 == v5 + 1 )
  {
    v32 = 0;
  }
  else
  {
    if ( *(_BYTE *)(v5 + 1) == 76 )
    {
      *(_QWORD *)a1 = v5 + 2;
      if ( (unsigned __int8)sub_E0EF70((_QWORD *)a1, (__int64 *)&v31) )
        return 0;
      v15 = *(_BYTE **)a1;
      v16 = *(_BYTE **)(a1 + 8);
      ++v31;
      if ( v15 == v16 || *v15 != 95 )
        return 0;
      v9 = v15 + 1;
      v32 = 0;
      *(_QWORD *)a1 = v9;
      if ( v16 == v9 )
        goto LABEL_9;
    }
    else
    {
      v32 = 0;
    }
    if ( *v9 == 95 )
    {
      *(_QWORD *)a1 = v9 + 1;
      goto LABEL_13;
    }
  }
LABEL_9:
  if ( (unsigned __int8)sub_E0EF70((_QWORD *)a1, (__int64 *)&v32) )
    return 0;
  ++v32;
  v11 = *(_BYTE **)a1;
  if ( *(_QWORD *)a1 == *(_QWORD *)(a1 + 8) || *v11 != 95 )
    return 0;
  *(_QWORD *)a1 = v11 + 1;
LABEL_13:
  if ( *(_BYTE *)(a1 + 778) )
    return sub_E0FB60(a1 + 816, *(_QWORD *)a1 + ~v5, v5, a4, a5, v5);
  if ( *(_BYTE *)(a1 + 777) && !v31 )
  {
    v17 = v32;
    v18 = sub_E0E790(a1 + 816, 40, 0, a4, a5, v5);
    v7 = v18;
    if ( !v18 )
      return 0;
    *(_QWORD *)(v18 + 16) = v17;
    *(_WORD *)(v18 + 8) = -32724;
    v22 = *(_BYTE *)(v18 + 10);
    *(_QWORD *)(v7 + 24) = 0;
    *(_BYTE *)(v7 + 32) = 0;
    *(_BYTE *)(v7 + 10) = v22 & 0xF0 | 0xA;
    *(_QWORD *)v7 = &unk_49DFE48;
    v23 = *(__int64 **)(a1 + 728);
    if ( v23 != *(__int64 **)(a1 + 736) )
    {
LABEL_36:
      *(_QWORD *)(a1 + 728) = v23 + 1;
      *v23 = v7;
      return v7;
    }
    v24 = *(_QWORD *)(a1 + 720);
    v25 = (size_t)v23 - v24;
    v26 = 16 * (((__int64)v23 - v24) >> 3);
    if ( v24 == a1 + 744 )
    {
      src = *(__int64 **)(a1 + 720);
      v29 = (void *)malloc(16 * (((__int64)v23 - v24) >> 3), 40, v19, v24, v20, v21);
      if ( v29 )
      {
        if ( v23 != src )
          v29 = memmove(v29, src, v25);
        *(_QWORD *)(a1 + 720) = v29;
        goto LABEL_39;
      }
    }
    else
    {
      v27 = realloc(*(void **)(a1 + 720));
      *(_QWORD *)(a1 + 720) = v27;
      if ( v27 )
      {
LABEL_39:
        v28 = (__int64 *)(*(_QWORD *)(a1 + 720) + v25);
        *(_QWORD *)(a1 + 736) = *(_QWORD *)(a1 + 720) + v26;
        v23 = v28;
        goto LABEL_36;
      }
    }
    abort();
  }
  v12 = *(_QWORD *)(a1 + 664);
  v13 = (*(_QWORD *)(a1 + 672) - v12) >> 3;
  if ( v13 <= v31 )
  {
    if ( *(_QWORD *)(a1 + 784) != v31 || v13 != v31 )
      return 0;
    v33[0] = 0;
    sub_E18730(a1 + 664, v33, v31, v12, a5, v5);
    return sub_E0FD70(a1 + 816, "auto");
  }
  v14 = *(_QWORD **)(v12 + 8 * v31);
  if ( v14 && v32 < (__int64)(v14[1] - *v14) >> 3 )
    return *(_QWORD *)(*v14 + 8 * v32);
  if ( v31 == *(_QWORD *)(a1 + 784) )
    return sub_E0FD70(a1 + 816, "auto");
  return 0;
}
