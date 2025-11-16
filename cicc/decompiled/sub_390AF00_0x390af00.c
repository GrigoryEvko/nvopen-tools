// Function: sub_390AF00
// Address: 0x390af00
//
__int64 __fastcall sub_390AF00(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  _QWORD *v5; // rdx
  _QWORD *v6; // rax
  _QWORD *v7; // r13
  __int64 v8; // rax
  unsigned int v9; // r13d
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rdi
  __int64 *v14; // r15
  __int64 *v15; // rax
  _QWORD *v16; // rdx
  __int64 *v17; // rcx
  unsigned int v18; // esi
  __int64 v19; // [rsp+0h] [rbp-50h] BYREF
  __int64 *v20; // [rsp+8h] [rbp-48h]
  __int64 v21; // [rsp+10h] [rbp-40h]
  int v22; // [rsp+18h] [rbp-38h]

  v2 = a1 + 184;
  v5 = *(_QWORD **)(a1 + 200);
  v6 = *(_QWORD **)(a1 + 192);
  if ( v5 == v6 )
  {
    v7 = &v6[*(unsigned int *)(a1 + 212)];
    if ( v6 == v7 )
    {
      v16 = *(_QWORD **)(a1 + 192);
    }
    else
    {
      do
      {
        if ( a2 == *v6 )
          break;
        ++v6;
      }
      while ( v7 != v6 );
      v16 = v7;
    }
  }
  else
  {
    v7 = &v5[*(unsigned int *)(a1 + 208)];
    v6 = sub_16CC9F0(a1 + 184, a2);
    if ( a2 == *v6 )
    {
      v11 = *(_QWORD *)(a1 + 200);
      if ( v11 == *(_QWORD *)(a1 + 192) )
        v12 = *(unsigned int *)(a1 + 212);
      else
        v12 = *(unsigned int *)(a1 + 208);
      v16 = (_QWORD *)(v11 + 8 * v12);
    }
    else
    {
      v8 = *(_QWORD *)(a1 + 200);
      if ( v8 != *(_QWORD *)(a1 + 192) )
      {
        v6 = (_QWORD *)(v8 + 8LL * *(unsigned int *)(a1 + 208));
        goto LABEL_5;
      }
      v6 = (_QWORD *)(v8 + 8LL * *(unsigned int *)(a1 + 212));
      v16 = v6;
    }
  }
  while ( v16 != v6 && *v6 >= 0xFFFFFFFFFFFFFFFELL )
    ++v6;
LABEL_5:
  if ( v6 != v7 )
    return 1;
  if ( (*(_BYTE *)(a2 + 9) & 0xC) != 8 )
    return 0;
  *(_BYTE *)(a2 + 8) |= 4u;
  v13 = *(_QWORD *)(a2 + 24);
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  if ( !(unsigned __int8)sub_38CF2C0(v13, (__int64)&v19, 0, 0) )
    return 0;
  v14 = v20;
  if ( v20 )
    return 0;
  if ( v22 )
    return 0;
  if ( !v19 )
    return 0;
  if ( *(_WORD *)(v19 + 16) )
    return 0;
  v9 = sub_390AF00(a1, *(_QWORD *)(v19 + 24));
  if ( !(_BYTE)v9 )
    return 0;
  v15 = *(__int64 **)(a1 + 192);
  if ( *(__int64 **)(a1 + 200) != v15 )
  {
LABEL_29:
    sub_16CCBA0(v2, a2);
    return v9;
  }
  v17 = &v15[*(unsigned int *)(a1 + 212)];
  v18 = *(_DWORD *)(a1 + 212);
  if ( v15 == v17 )
  {
LABEL_38:
    if ( v18 < *(_DWORD *)(a1 + 208) )
    {
      *(_DWORD *)(a1 + 212) = v18 + 1;
      *v17 = a2;
      ++*(_QWORD *)(a1 + 184);
      return v9;
    }
    goto LABEL_29;
  }
  while ( a2 != *v15 )
  {
    if ( *v15 == -2 )
      v14 = v15;
    if ( v17 == ++v15 )
    {
      if ( !v14 )
        goto LABEL_38;
      *v14 = a2;
      --*(_DWORD *)(a1 + 216);
      ++*(_QWORD *)(a1 + 184);
      return v9;
    }
  }
  return v9;
}
