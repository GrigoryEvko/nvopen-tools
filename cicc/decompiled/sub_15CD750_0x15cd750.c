// Function: sub_15CD750
// Address: 0x15cd750
//
__int64 __fastcall sub_15CD750(__int64 a1, unsigned __int8 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // r15
  int v9; // r13d
  unsigned int v10; // ebx
  unsigned int v11; // r14d
  bool v12; // dl
  bool v13; // cl
  bool v14; // al
  int v15; // eax
  unsigned __int64 v16; // r12
  __int64 v17; // r15
  __int64 v18; // r12
  __int64 *v19; // rdi
  __int64 v20; // rcx
  __int64 *v21; // rdx
  __int64 v22; // rax
  int v25; // [rsp+18h] [rbp-38h]

  result = 0;
  if ( a3 == a4 )
    return result;
  v7 = sub_157EBA0(a3);
  if ( !v7 )
    goto LABEL_23;
  v25 = sub_15F4D60(v7);
  v8 = sub_157EBA0(a3);
  v9 = v25 >> 2;
  if ( v25 >> 2 <= 0 )
  {
    v15 = v25;
    v10 = 0;
LABEL_19:
    if ( v15 != 2 )
    {
      if ( v15 != 3 )
      {
        if ( v15 != 1 )
          goto LABEL_23;
        goto LABEL_22;
      }
      if ( a4 == sub_15F4DF0(v8, v10) )
        goto LABEL_10;
      ++v10;
    }
    if ( a4 == sub_15F4DF0(v8, v10) )
      goto LABEL_10;
    ++v10;
LABEL_22:
    if ( a4 == sub_15F4DF0(v8, v10) )
      goto LABEL_10;
LABEL_23:
    v14 = a2 == 0;
    if ( !a2 )
      return 0;
    goto LABEL_24;
  }
  v10 = 0;
  while ( a4 != sub_15F4DF0(v8, v10) )
  {
    v11 = v10 + 1;
    if ( a4 == sub_15F4DF0(v8, v10 + 1)
      || (v11 = v10 + 2, a4 == sub_15F4DF0(v8, v10 + 2))
      || (v11 = v10 + 3, a4 == sub_15F4DF0(v8, v10 + 3)) )
    {
      v12 = v25 == v11;
      v13 = v25 != v11;
      v14 = a2 == 0;
      if ( a2 )
        goto LABEL_11;
      goto LABEL_16;
    }
    v10 += 4;
    if ( !--v9 )
    {
      v15 = v25 - v10;
      goto LABEL_19;
    }
  }
LABEL_10:
  v12 = v25 == v10;
  v13 = v25 != v10;
  v14 = a2 == 0;
  if ( !a2 )
  {
LABEL_16:
    if ( v12 )
      return 0;
  }
LABEL_11:
  if ( a2 == 1 && v13 )
    return 0;
LABEL_24:
  v16 = a4 & 0xFFFFFFFFFFFFFFFBLL;
  v17 = v16 | (4LL * a2);
  v18 = (4LL * v14) | v16;
  v19 = *(__int64 **)(a1 + 8);
  v20 = *(unsigned int *)(a1 + 16);
  v21 = &v19[2 * v20];
  if ( v19 == v21 )
  {
LABEL_27:
    if ( (unsigned int)v20 >= *(_DWORD *)(a1 + 20) )
    {
      sub_16CD150(a1 + 8, a1 + 24, 0, 16);
      v21 = (__int64 *)(*(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16));
    }
    v21[1] = v17;
    *v21 = a3;
    ++*(_DWORD *)(a1 + 16);
    return 1;
  }
  else
  {
    while ( 1 )
    {
      if ( a3 == *v19 )
      {
        v22 = v19[1];
        if ( v17 == v22 )
          return 0;
        if ( v18 == v22 )
          break;
      }
      v19 += 2;
      if ( v21 == v19 )
        goto LABEL_27;
    }
    if ( v21 != v19 + 2 )
    {
      memmove(v19, v19 + 2, (char *)v21 - (char *)(v19 + 2));
      LODWORD(v20) = *(_DWORD *)(a1 + 16);
    }
    *(_DWORD *)(a1 + 16) = v20 - 1;
    return 0;
  }
}
