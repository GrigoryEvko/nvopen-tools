// Function: sub_222AC50
// Address: 0x222ac50
//
__int64 __fastcall sub_222AC50(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  wint_t v5; // eax
  int v6; // eax

  v3 = 0;
  if ( a3 )
  {
    while ( 1 )
    {
      v5 = getwc(*(__FILE **)(a1 + 64));
      if ( v5 == -1 )
        break;
      *(_DWORD *)(a2 + 4 * v3++) = v5;
      if ( a3 == v3 )
        goto LABEL_7;
    }
    if ( !v3 )
      goto LABEL_6;
LABEL_7:
    v6 = *(_DWORD *)(a2 + 4 * v3 - 4);
  }
  else
  {
LABEL_6:
    v3 = 0;
    v6 = -1;
  }
  *(_DWORD *)(a1 + 72) = v6;
  return v3;
}
