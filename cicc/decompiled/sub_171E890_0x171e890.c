// Function: sub_171E890
// Address: 0x171e890
//
__int64 __fastcall sub_171E890(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  int v4; // eax
  int v5; // eax
  __int64 *v6; // rsi
  __int64 v7; // rdx
  unsigned __int64 v8; // rax
  __int64 v9; // rsi
  int v10; // eax
  __int64 *v11; // rax
  __int64 v12; // rax
  _QWORD *v13; // rcx
  _QWORD **v14; // rdi
  __int64 v15; // rdx
  unsigned __int64 v16; // rcx
  void *v17; // rsi

  v2 = *(_QWORD *)(a2 + 8);
  if ( !v2 || *(_QWORD *)(v2 + 8) )
    return 0;
  v4 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)v4 <= 0x17u )
  {
    if ( (_BYTE)v4 != 5 )
      return 0;
    v5 = *(unsigned __int16 *)(a2 + 18);
  }
  else
  {
    v5 = v4 - 24;
  }
  if ( v5 != 37 )
    return 0;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v6 = *(__int64 **)(a2 - 8);
  else
    v6 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v7 = *v6;
  v8 = *(unsigned __int8 *)(*v6 + 16);
  if ( (unsigned __int8)v8 <= 0x17u )
  {
    if ( (_BYTE)v8 == 5 )
    {
      v16 = *(unsigned __int16 *)(v7 + 18);
      if ( (unsigned __int16)v16 <= 0x17u )
      {
        v17 = &loc_80A800;
        v10 = (unsigned __int16)v16;
        if ( _bittest64((const __int64 *)&v17, v16) )
          goto LABEL_13;
      }
    }
    return 0;
  }
  if ( (unsigned __int8)v8 > 0x2Fu )
    return 0;
  v9 = 0x80A800000000LL;
  if ( !_bittest64(&v9, v8) )
    return 0;
  v10 = (unsigned __int8)v8 - 24;
LABEL_13:
  if ( v10 != 11 || (*(_BYTE *)(v7 + 17) & 2) == 0 )
    return 0;
  v11 = (*(_BYTE *)(v7 + 23) & 0x40) != 0
      ? *(__int64 **)(v7 - 8)
      : (__int64 *)(v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF));
  v12 = *v11;
  if ( !v12 )
    return 0;
  v13 = *a1;
  v14 = a1 + 1;
  *v13 = v12;
  if ( (*(_BYTE *)(v7 + 23) & 0x40) != 0 )
    v15 = *(_QWORD *)(v7 - 8);
  else
    v15 = v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF);
  return sub_13D2630(v14, *(_BYTE **)(v15 + 24));
}
