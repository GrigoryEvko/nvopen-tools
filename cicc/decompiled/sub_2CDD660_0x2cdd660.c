// Function: sub_2CDD660
// Address: 0x2cdd660
//
__int64 __fastcall sub_2CDD660(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  _DWORD *v5; // rdx
  _DWORD *v6; // rsi
  _DWORD *v7; // rax
  _DWORD *v8; // rcx

  v5 = *(_DWORD **)(a2 + 72);
  v6 = &v5[*(unsigned int *)(a2 + 80)];
  v7 = *(_DWORD **)(a1 + 72);
  v8 = &v7[*(unsigned int *)(a1 + 80)];
  LOBYTE(a5) = v6 == v5 || v8 == v7;
  if ( (_BYTE)a5 )
  {
LABEL_7:
    LOBYTE(a1) = v8 != v7;
    LOBYTE(v7) = v6 != v5;
    return ((unsigned int)v7 | (unsigned int)a1) ^ 1;
  }
  else
  {
    while ( 1 )
    {
      LODWORD(a1) = *v5;
      if ( *v7 != *v5 )
        return a5;
      ++v7;
      ++v5;
      if ( v7 == v8 || v5 == v6 )
        goto LABEL_7;
    }
  }
}
