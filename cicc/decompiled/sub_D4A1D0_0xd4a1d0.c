// Function: sub_D4A1D0
// Address: 0xd4a1d0
//
__int16 __fastcall sub_D4A1D0(__int64 a1, const void *a2, size_t a3, __int64 a4, __int64 a5, __int64 a6)
{
  _BYTE *v6; // rax
  unsigned __int8 v7; // dl
  int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax

  v6 = sub_D49780(a1, a2, a3, a4, a5, a6);
  if ( !v6 )
  {
    BYTE1(v6) = 0;
    return (__int16)v6;
  }
  v7 = *(v6 - 16);
  if ( (v7 & 2) != 0 )
  {
    v8 = *((_DWORD *)v6 - 6);
    if ( v8 == 1 )
    {
LABEL_4:
      LOWORD(v6) = 257;
      return (__int16)v6;
    }
    if ( v8 == 2 )
    {
      v9 = *((_QWORD *)v6 - 4);
      goto LABEL_8;
    }
LABEL_17:
    BUG();
  }
  if ( ((*((_WORD *)v6 - 8) >> 6) & 0xF) == 1 )
    goto LABEL_4;
  if ( ((*((_WORD *)v6 - 8) >> 6) & 0xF) != 2 )
    goto LABEL_17;
  v9 = (__int64)&v6[-8 * ((v7 >> 2) & 0xF) - 16];
LABEL_8:
  v10 = *(_QWORD *)(v9 + 8);
  if ( !v10 )
    goto LABEL_4;
  v11 = *(_QWORD *)(v10 + 136);
  if ( !v11 )
    goto LABEL_4;
  if ( *(_DWORD *)(v11 + 32) <= 0x40u )
    v6 = *(_BYTE **)(v11 + 24);
  else
    v6 = **(_BYTE ***)(v11 + 24);
  LOBYTE(v6) = v6 != 0;
  BYTE1(v6) = 1;
  return (__int16)v6;
}
