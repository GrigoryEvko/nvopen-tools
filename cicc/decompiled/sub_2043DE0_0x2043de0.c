// Function: sub_2043DE0
// Address: 0x2043de0
//
__int64 __fastcall sub_2043DE0(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // dl
  __int64 result; // rax
  unsigned __int16 v4; // dx
  __int64 v5; // rcx
  char v6; // dl
  unsigned int v7; // edx

  v2 = *(_BYTE *)(a2 + 16);
  result = a1;
  if ( v2 <= 0x17u )
    goto LABEL_4;
  if ( v2 == 25 )
  {
    v4 = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 56LL) + 18LL);
    *(_BYTE *)(a1 + 4) = 1;
    *(_DWORD *)a1 = (v4 >> 4) & 0x3FF;
  }
  else
  {
    if ( v2 != 78 )
    {
LABEL_4:
      *(_BYTE *)(a1 + 4) = 0;
      return result;
    }
    v5 = *(_QWORD *)(a2 - 24);
    v6 = *(_BYTE *)(v5 + 16);
    if ( v6 )
    {
      if ( v6 == 20 )
        goto LABEL_4;
    }
    else if ( *(_DWORD *)(v5 + 36) )
    {
      goto LABEL_4;
    }
    v7 = *(unsigned __int16 *)(a2 + 18);
    *(_BYTE *)(a1 + 4) = 1;
    *(_DWORD *)a1 = (v7 >> 2) & 0x3FFFDFFF;
  }
  return result;
}
