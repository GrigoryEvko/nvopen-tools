// Function: sub_103E5A0
// Address: 0x103e5a0
//
bool __fastcall sub_103E5A0(__int64 a1, unsigned __int8 *a2)
{
  unsigned __int8 *v2; // r12
  unsigned __int8 v3; // al
  unsigned __int8 v4; // dl
  bool result; // al
  unsigned __int8 *v6; // rax
  unsigned __int8 *v7; // rax

  v2 = sub_BD3990(a2, (__int64)a2);
  v3 = *v2;
  if ( *v2 <= 0x1Cu )
    goto LABEL_9;
  if ( sub_AA5B70(*((_QWORD *)v2 + 5)) )
    return 1;
  v3 = *v2;
  if ( *v2 <= 0x1Cu )
  {
LABEL_9:
    if ( v3 != 5 || *((_WORD *)v2 + 1) != 34 )
      goto LABEL_5;
  }
  else if ( v3 != 63 )
  {
LABEL_5:
    v4 = *sub_BD3990(v2, (__int64)a2);
    return v4 == 60 || v4 <= 0x1Cu;
  }
  v6 = sub_BD3990(*(unsigned __int8 **)&v2[-32 * (*((_DWORD *)v2 + 1) & 0x7FFFFFF)], (__int64)a2);
  result = *v6 == 60 || *v6 <= 0x1Cu;
  if ( result )
  {
    v7 = &v2[32 * (1LL - (*((_DWORD *)v2 + 1) & 0x7FFFFFF))];
    if ( v2 == v7 )
      return 1;
    while ( **(_BYTE **)v7 == 17 )
    {
      v7 += 32;
      if ( v2 == v7 )
        return 1;
    }
    return 0;
  }
  return result;
}
