// Function: sub_2576450
// Address: 0x2576450
//
char __fastcall sub_2576450(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r8
  unsigned __int64 v3; // rax
  int v4; // edx
  char v5; // dl
  __int64 v6; // rcx

  v2 = sub_2566C40(a2 + 32, (__int64 *)(a1 + 72));
  LOBYTE(v3) = *(_BYTE *)(a1 + 104);
  if ( v2 )
  {
    *(_BYTE *)(a1 + 105) = v3;
    return v3;
  }
  if ( *(_BYTE *)(a1 + 105) == (_BYTE)v3 )
    return v3;
  v3 = sub_250D070((_QWORD *)(a1 + 72));
  v4 = *(unsigned __int8 *)v3;
  if ( (_BYTE)v4 == 17 )
  {
    v5 = *(_BYTE *)(a1 + 105);
    if ( v5 )
    {
      sub_2575FB0((_DWORD *)(a1 + 112), (const void **)(v3 + 24));
      LODWORD(v3) = *(_DWORD *)(a1 + 152);
      if ( (unsigned int)v3 < unk_4FEF868 )
      {
        v5 = *(_BYTE *)(a1 + 105);
        LOBYTE(v3) = (_DWORD)v3 == 0;
        *(_BYTE *)(a1 + 288) &= v3;
      }
      else
      {
        v5 = *(_BYTE *)(a1 + 104);
        *(_BYTE *)(a1 + 105) = v5;
      }
    }
    *(_BYTE *)(a1 + 104) = v5;
  }
  else if ( (unsigned int)(unsigned __int8)v4 - 12 > 1 )
  {
    if ( (unsigned __int8)v4 <= 0x1Cu || (v3 = (unsigned int)(v4 - 42), (unsigned __int8)(v4 - 42) > 0x28u) )
    {
      LOBYTE(v3) = v4 & 0xFD;
      if ( (v4 & 0xFD) == 0x54 )
        return v3;
    }
    else
    {
      v6 = 0x13FFE03FFFFLL;
      if ( _bittest64(&v6, v3) )
        return v3;
    }
    if ( (_BYTE)v4 != 61 )
    {
      LOBYTE(v3) = *(_BYTE *)(a1 + 104);
      *(_BYTE *)(a1 + 105) = v3;
    }
  }
  else
  {
    LOBYTE(v3) = *(_BYTE *)(a1 + 105);
    *(_BYTE *)(a1 + 288) = *(_DWORD *)(a1 + 152) == 0;
    *(_BYTE *)(a1 + 104) = v3;
  }
  return v3;
}
