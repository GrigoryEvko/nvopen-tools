// Function: sub_1E1A450
// Address: 0x1e1a450
//
__int64 __fastcall sub_1E1A450(__int64 a1, int a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v6; // rcx
  int v7; // r10d
  __int64 v8; // r9
  __int64 v9; // rbx
  unsigned int v10; // r8d
  __int16 v11; // dx
  _WORD *v12; // r8
  unsigned __int16 v13; // dx
  _WORD *v14; // r12
  _WORD *v15; // r13
  unsigned int v16; // r9d
  unsigned __int16 *v17; // rbx
  int v18; // r8d
  unsigned int i; // r10d
  bool v20; // cf
  int v21; // r10d

  if ( a2 <= 0 )
    a3 = 0;
  result = *(_QWORD *)(a1 + 32);
  v6 = result + 40LL * *(unsigned int *)(a1 + 40);
  if ( result == v6 )
    return result;
  while ( 1 )
  {
    if ( *(_BYTE *)result || (*(_BYTE *)(result + 3) & 0x10) != 0 || (*(_BYTE *)(result + 3) & 0x40) == 0 )
      goto LABEL_27;
    v7 = *(_DWORD *)(result + 8);
    if ( a3 )
      break;
    if ( a2 == v7 )
      goto LABEL_26;
LABEL_27:
    result += 40;
    if ( v6 == result )
      return result;
  }
  if ( a2 == v7 )
  {
LABEL_26:
    *(_BYTE *)(result + 3) &= ~0x40u;
    goto LABEL_27;
  }
  if ( a2 < 0 || v7 < 0 )
    goto LABEL_27;
LABEL_11:
  v8 = *(_QWORD *)(a3 + 8);
  v9 = *(_QWORD *)(a3 + 56);
  v10 = *(_DWORD *)(v8 + 24LL * (unsigned int)a2 + 16);
  v11 = a2 * (v10 & 0xF);
  v12 = (_WORD *)(v9 + 2LL * (v10 >> 4));
  v13 = *v12 + v11;
  v14 = v12 + 1;
  LODWORD(v8) = *(_DWORD *)(v8 + 24LL * (unsigned int)v7 + 16);
  v15 = (_WORD *)(v9 + 2LL * ((unsigned int)v8 >> 4));
  LOBYTE(v12) = v8;
  v16 = v13;
  v17 = v15 + 1;
  v18 = v7 * ((unsigned __int8)v12 & 0xF);
  LOWORD(v18) = *v15 + v18;
  for ( i = (unsigned __int16)v18; ; i = (unsigned __int16)v18 )
  {
    v20 = v16 < i;
    if ( v16 == i )
      break;
    while ( v20 )
    {
      v13 += *v14;
      if ( !*v14 )
        goto LABEL_17;
      v16 = v13;
      ++v14;
      v20 = v13 < i;
      if ( v13 == i )
        goto LABEL_16;
    }
    v21 = *v17;
    if ( !(_WORD)v21 )
      goto LABEL_17;
    v18 += v21;
    ++v17;
  }
LABEL_16:
  *(_BYTE *)(result + 3) &= ~0x40u;
LABEL_17:
  while ( 1 )
  {
    result += 40;
    if ( v6 == result )
      return result;
    while ( !*(_BYTE *)result && (*(_BYTE *)(result + 3) & 0x10) == 0 && (*(_BYTE *)(result + 3) & 0x40) != 0 )
    {
      v7 = *(_DWORD *)(result + 8);
      if ( a2 == v7 )
        goto LABEL_16;
      if ( v7 >= 0 )
        goto LABEL_11;
      result += 40;
      if ( v6 == result )
        return result;
    }
  }
}
