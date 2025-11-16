// Function: sub_1E1D5E0
// Address: 0x1e1d5e0
//
__int64 __fastcall sub_1E1D5E0(__int64 a1)
{
  __int64 result; // rax
  unsigned __int16 *v2; // rdx
  unsigned __int16 *v3; // rax
  int v4; // edx
  _QWORD *v5; // rcx
  __int64 v6; // rsi
  __int64 v7; // rdx
  _QWORD *v8; // rcx
  bool v9; // zf
  _DWORD *v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rdx

  LODWORD(result) = *(unsigned __int16 *)(a1 + 48);
  while ( 1 )
  {
    v2 = *(unsigned __int16 **)(a1 + 56);
    *(_QWORD *)(a1 + 56) = v2 + 1;
    LODWORD(v2) = *v2;
    result = (unsigned int)((_DWORD)v2 + result);
    *(_WORD *)(a1 + 48) = result;
    if ( (_WORD)v2 )
      goto LABEL_7;
    result = *(unsigned __int16 *)(a1 + 42);
    *(_QWORD *)(a1 + 56) = 0;
    *(_DWORD *)(a1 + 40) = (unsigned __int16)result;
    if ( !(_WORD)result )
      break;
    v5 = *(_QWORD **)(a1 + 8);
    v6 = *(unsigned int *)(*v5 + 24LL * (unsigned __int16)result + 8);
    v7 = v5[6];
    *(_WORD *)(a1 + 48) = result;
    *(_QWORD *)(a1 + 56) = v7 + 2 * v6;
LABEL_7:
    if ( *(_BYTE *)(a1 + 16) || !*(_QWORD *)(a1 + 32) )
      return result;
    result = *(unsigned __int16 *)(a1 + 48);
LABEL_10:
    if ( *(_DWORD *)a1 != (unsigned __int16)result )
      return result;
  }
  v3 = *(unsigned __int16 **)(a1 + 32);
  *(_QWORD *)(a1 + 32) = v3 + 1;
  v4 = *v3;
  result = v4 + (unsigned int)*(unsigned __int16 *)(a1 + 24);
  *(_WORD *)(a1 + 24) += v4;
  if ( (_WORD)v4 )
  {
    v8 = *(_QWORD **)(a1 + 8);
    v9 = *(_BYTE *)(a1 + 16) == 0;
    v10 = (_DWORD *)(v8[5] + 4LL * (unsigned __int16)result);
    result = *(unsigned __int16 *)v10;
    *(_DWORD *)(a1 + 40) = *v10;
    v11 = *(unsigned int *)(*v8 + 24LL * (unsigned __int16)result + 8);
    v12 = v8[6];
    *(_WORD *)(a1 + 48) = result;
    *(_QWORD *)(a1 + 56) = v12 + 2 * v11;
    if ( !v9 )
      return result;
    goto LABEL_10;
  }
  *(_QWORD *)(a1 + 32) = 0;
  return result;
}
