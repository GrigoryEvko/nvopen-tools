// Function: sub_386B050
// Address: 0x386b050
//
_QWORD *__fastcall sub_386B050(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rsi
  __int64 v6; // r13
  char v7; // bl
  unsigned int v8; // r12d
  __int64 v9; // rax
  char v10; // cl
  __int64 v11; // r9
  int v12; // edx
  __int64 v13; // r11
  __int64 v14; // rcx
  _QWORD *i; // rcx
  __int64 v16; // r9
  _QWORD *result; // rax
  _QWORD *v18; // rax
  __int64 v19; // r9
  unsigned __int64 v20; // rsi
  __int64 v21; // rsi

  v4 = *(unsigned int *)(a1 + 76);
  v6 = 24 * v4;
  v7 = *(_BYTE *)(a1 + 23);
  v8 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  v9 = v8;
  v10 = v7 & 0x40;
  if ( v8 )
  {
    v11 = v6 + 8;
    v12 = 0;
    while ( 1 )
    {
      v13 = a1 - 24LL * v8;
      if ( v10 )
        v13 = *(_QWORD *)(a1 - 8);
      if ( a2 == *(_QWORD *)(v13 + v11) )
        break;
      ++v12;
      v11 += 8;
      if ( v8 == v12 )
        goto LABEL_9;
    }
    v6 += 8 + 8LL * v12;
    if ( !v10 )
      goto LABEL_8;
LABEL_10:
    v14 = *(_QWORD *)(a1 - 8);
    goto LABEL_11;
  }
LABEL_9:
  v12 = -1;
  if ( v10 )
    goto LABEL_10;
LABEL_8:
  v14 = a1 - 24LL * v8;
LABEL_11:
  for ( i = (_QWORD *)(v6 + v14); ; ++i )
  {
    v16 = (v7 & 0x40) != 0 ? *(_QWORD *)(a1 - 8) : a1 - 24 * v9;
    result = (_QWORD *)(v16 + 24 * v4 + 8 * v9 + 8);
    if ( i == result || *i != a2 )
      break;
    v18 = (_QWORD *)(v16 + 24LL * (unsigned int)v12);
    if ( *v18 )
    {
      v19 = v18[1];
      v20 = v18[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v20 = v19;
      if ( v19 )
        *(_QWORD *)(v19 + 16) = *(_QWORD *)(v19 + 16) & 3LL | v20;
    }
    *v18 = a3;
    if ( a3 )
    {
      v21 = *(_QWORD *)(a3 + 8);
      v18[1] = v21;
      if ( v21 )
        *(_QWORD *)(v21 + 16) = (unsigned __int64)(v18 + 1) | *(_QWORD *)(v21 + 16) & 3LL;
      v18[2] = (a3 + 8) | v18[2] & 3LL;
      *(_QWORD *)(a3 + 8) = v18;
    }
    v7 = *(_BYTE *)(a1 + 23);
    ++v12;
    v4 = *(unsigned int *)(a1 + 76);
    v9 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  }
  return result;
}
