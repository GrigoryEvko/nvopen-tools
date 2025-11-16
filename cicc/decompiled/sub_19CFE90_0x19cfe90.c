// Function: sub_19CFE90
// Address: 0x19cfe90
//
__int64 __fastcall sub_19CFE90(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // r13
  unsigned __int8 v9; // al
  __int64 v10; // r15
  unsigned __int8 v11; // al
  __int64 *v12; // rax
  bool v14; // zf
  char v15; // al
  char v16; // al
  __int64 *v17; // rax
  _QWORD *v18; // rdi
  _QWORD *v19; // rdx
  int v20; // ecx
  unsigned int i; // r12d
  __int64 v22; // r15
  __int64 v23; // rax
  _BYTE v24[49]; // [rsp+1Fh] [rbp-31h] BYREF

  v6 = sub_1649C60(a1);
  v7 = sub_1649C60(a2);
  if ( v6 == v7 )
  {
    *a3 = 0;
    return 1;
  }
  v8 = v7;
  v9 = *(_BYTE *)(v6 + 16);
  if ( v9 > 0x17u )
  {
    v14 = v9 == 56;
    v11 = *(_BYTE *)(v8 + 16);
    v10 = 0;
    if ( v14 )
      v10 = v6;
    if ( v11 > 0x17u )
    {
LABEL_7:
      if ( v11 != 56 )
        goto LABEL_8;
LABEL_18:
      v15 = *(_BYTE *)(v8 + 23);
      v24[0] = 0;
      v16 = v15 & 0x40;
      if ( v10 )
      {
        if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
          v18 = *(_QWORD **)(v10 - 8);
        else
          v18 = (_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
        if ( v16 )
          v19 = *(_QWORD **)(v8 - 8);
        else
          v19 = (_QWORD *)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF));
        if ( *v19 != *v18 )
          return 0;
        v20 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
        if ( v20 == 1 )
        {
          i = 1;
        }
        else
        {
          for ( i = 1; i != v20; ++i )
          {
            if ( (*(_DWORD *)(v8 + 20) & 0xFFFFFFF) == i )
              break;
            if ( v19[3 * i] != v18[3 * i] )
              break;
          }
        }
        v22 = sub_19CF890(v10, i, v24, a4);
        v23 = sub_19CF890(v8, i, v24, a4);
        if ( v24[0] )
          return 0;
        *a3 = v23 - v22;
        return 1;
      }
      else
      {
        if ( v16 )
          v17 = *(__int64 **)(v8 - 8);
        else
          v17 = (__int64 *)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF));
        if ( v6 != sub_1649C60(*v17) )
          return 0;
        *a3 = sub_19CF890(v8, 1, v24, a4);
        return v24[0] ^ 1u;
      }
    }
  }
  else
  {
    v10 = 0;
    if ( v9 == 5 && *(_WORD *)(v6 + 18) == 32 )
      v10 = v6;
    v11 = *(_BYTE *)(v8 + 16);
    if ( v11 > 0x17u )
      goto LABEL_7;
  }
  if ( v11 == 5 && *(_WORD *)(v8 + 18) == 32 )
    goto LABEL_18;
LABEL_8:
  v24[0] = 0;
  if ( !v10 )
    return 0;
  v12 = (*(_BYTE *)(v10 + 23) & 0x40) != 0
      ? *(__int64 **)(v10 - 8)
      : (__int64 *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
  if ( v8 != sub_1649C60(*v12) )
    return 0;
  *a3 = -sub_19CF890(v10, 1, v24, a4);
  return v24[0] ^ 1u;
}
