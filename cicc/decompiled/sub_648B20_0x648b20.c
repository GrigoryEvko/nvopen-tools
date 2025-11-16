// Function: sub_648B20
// Address: 0x648b20
//
__int64 __fastcall sub_648B20(_BYTE *a1)
{
  __int64 v1; // rcx
  char v2; // al
  __int64 result; // rax
  int v4; // edx

  v1 = *(_QWORD *)a1;
  v2 = *(_BYTE *)(*(_QWORD *)a1 + 80LL);
  if ( v2 == 7 )
  {
LABEL_4:
    result = *(_QWORD *)(v1 + 88);
    goto LABEL_5;
  }
  if ( v2 != 21 )
  {
    if ( v2 != 9 )
      sub_721090(a1);
    goto LABEL_4;
  }
  result = *(_QWORD *)(*(_QWORD *)(v1 + 88) + 192LL);
LABEL_5:
  if ( (a1[224] & 1) != 0 )
  {
    if ( *(_BYTE *)(result + 136) > 2u )
    {
      return sub_6851C0(1378, a1 + 48);
    }
    else
    {
      v4 = *(_DWORD *)(result + 140);
      if ( (a1[127] & 0x10) != 0 || (v4 & 1) != 0 )
        *(_DWORD *)(result + 140) = v4 | 1;
      else
        return sub_6854C0(1876, a1 + 48, *(_QWORD *)a1);
    }
  }
  else if ( (a1[127] & 0x10) == 0 && (*(_BYTE *)(result + 140) & 1) != 0 )
  {
    return sub_6853B0(7, 1876, a1 + 48, v1);
  }
  return result;
}
