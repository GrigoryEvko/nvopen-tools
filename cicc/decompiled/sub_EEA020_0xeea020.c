// Function: sub_EEA020
// Address: 0xeea020
//
__int64 __fastcall sub_EEA020(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
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
  __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v20; // r9
  char v21; // al
  __int64 *v22; // r13
  __int64 v23; // rcx
  size_t v24; // r15
  __int64 v25; // r14
  __int64 v26; // rax
  __int64 *v27; // r15
  void *v28; // rdi
  __int64 *src; // [rsp+8h] [rbp-58h]
  unsigned __int64 v30; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int64 v31; // [rsp+18h] [rbp-48h] BYREF
  __int64 v32[8]; // [rsp+20h] [rbp-40h] BYREF

  v5 = *(_QWORD *)a1;
  v6 = *(_QWORD *)(a1 + 8);
  if ( *(_QWORD *)a1 == v6 || *(_BYTE *)v5 != 84 )
    return 0;
  v9 = (_BYTE *)(v5 + 1);
  v30 = 0;
  *(_QWORD *)a1 = v5 + 1;
  if ( v6 == v5 + 1 )
  {
    v31 = 0;
  }
  else
  {
    if ( *(_BYTE *)(v5 + 1) == 76 )
    {
      *(_QWORD *)a1 = v5 + 2;
      if ( (unsigned __int8)sub_EE35F0((_QWORD *)a1, (__int64 *)&v30) )
        return 0;
      v15 = *(_BYTE **)a1;
      v16 = *(_BYTE **)(a1 + 8);
      ++v30;
      if ( v15 == v16 || *v15 != 95 )
        return 0;
      v9 = v15 + 1;
      v31 = 0;
      *(_QWORD *)a1 = v9;
      if ( v16 == v9 )
        goto LABEL_9;
    }
    else
    {
      v31 = 0;
    }
    if ( *v9 == 95 )
    {
      *(_QWORD *)a1 = v9 + 1;
      goto LABEL_13;
    }
  }
LABEL_9:
  if ( (unsigned __int8)sub_EE35F0((_QWORD *)a1, (__int64 *)&v31) )
    return 0;
  ++v31;
  v11 = *(_BYTE **)a1;
  if ( *(_QWORD *)a1 == *(_QWORD *)(a1 + 8) || *v11 != 95 )
    return 0;
  *(_QWORD *)a1 = v11 + 1;
LABEL_13:
  if ( *(_BYTE *)(a1 + 778) )
  {
    v32[1] = v5;
    v32[0] = *(_QWORD *)a1 + ~v5;
    return sub_EE6A90(a1 + 808, v32);
  }
  if ( *(_BYTE *)(a1 + 777) && !v30 )
  {
    v17 = v31;
    v7 = sub_CD1D40((__int64 *)(a1 + 808), 40, 3);
    *(_WORD *)(v7 + 8) = -32724;
    v21 = *(_BYTE *)(v7 + 10);
    *(_QWORD *)(v7 + 16) = v17;
    *(_BYTE *)(v7 + 32) = 0;
    *(_QWORD *)(v7 + 24) = 0;
    *(_BYTE *)(v7 + 10) = v21 & 0xF0 | 0xA;
    *(_QWORD *)v7 = &unk_49DFE48;
    v22 = *(__int64 **)(a1 + 728);
    *(_QWORD *)(a1 + 920) = v7;
    if ( v22 != *(__int64 **)(a1 + 736) )
    {
LABEL_35:
      *(_QWORD *)(a1 + 728) = v22 + 1;
      *v22 = v7;
      return v7;
    }
    v23 = *(_QWORD *)(a1 + 720);
    v24 = (size_t)v22 - v23;
    v25 = 16 * (((__int64)v22 - v23) >> 3);
    if ( v23 == a1 + 744 )
    {
      src = *(__int64 **)(a1 + 720);
      v28 = (void *)malloc(16 * (((__int64)v22 - v23) >> 3), 40, v18, v23, v19, v20);
      if ( v28 )
      {
        if ( v22 != src )
          v28 = memmove(v28, src, v24);
        *(_QWORD *)(a1 + 720) = v28;
        goto LABEL_38;
      }
    }
    else
    {
      v26 = realloc(*(void **)(a1 + 720));
      *(_QWORD *)(a1 + 720) = v26;
      if ( v26 )
      {
LABEL_38:
        v27 = (__int64 *)(*(_QWORD *)(a1 + 720) + v24);
        *(_QWORD *)(a1 + 736) = *(_QWORD *)(a1 + 720) + v25;
        v22 = v27;
        goto LABEL_35;
      }
    }
    abort();
  }
  v12 = *(_QWORD *)(a1 + 664);
  v13 = (*(_QWORD *)(a1 + 672) - v12) >> 3;
  if ( v13 <= v30 )
  {
    if ( *(_QWORD *)(a1 + 784) != v30 || v13 != v30 )
      return 0;
    v32[0] = 0;
    sub_E18730(a1 + 664, v32, v30, v12, a5, v5);
    return sub_EE68C0(a1 + 808, "auto");
  }
  v14 = *(_QWORD **)(v12 + 8 * v30);
  if ( v14 && v31 < (__int64)(v14[1] - *v14) >> 3 )
    return *(_QWORD *)(*v14 + 8 * v31);
  if ( v30 == *(_QWORD *)(a1 + 784) )
    return sub_EE68C0(a1 + 808, "auto");
  return 0;
}
