// Function: sub_8DD400
// Address: 0x8dd400
//
__int64 __fastcall sub_8DD400(__int64 a1, _DWORD *a2)
{
  char v2; // al
  __int64 result; // rax
  unsigned __int64 v4; // rax
  __int64 v5; // rdx

  v2 = *(_BYTE *)(a1 + 140);
  if ( (unsigned __int8)(v2 - 9) > 2u )
  {
    if ( *(char *)(a1 + 90) < 0 )
    {
LABEL_12:
      *a2 = 1;
      sub_875AD0(a1, dword_4F07508);
      return 1;
    }
    if ( v2 == 12 )
    {
      if ( *(_QWORD *)(a1 + 8) )
        goto LABEL_3;
      v4 = *(unsigned __int8 *)(a1 + 184);
      if ( (unsigned __int8)v4 <= 0xCu )
      {
        v5 = 6338;
        if ( _bittest64(&v5, v4) )
          goto LABEL_3;
      }
    }
  }
  else
  {
    if ( (*(_BYTE *)(*(_QWORD *)(a1 + 168) + 112LL) & 4) != 0 )
    {
LABEL_3:
      *a2 = 1;
      return 0;
    }
    if ( *(char *)(a1 + 90) < 0 )
      goto LABEL_12;
  }
  result = sub_8DD3B0(a1);
  if ( (_DWORD)result )
    goto LABEL_3;
  return result;
}
