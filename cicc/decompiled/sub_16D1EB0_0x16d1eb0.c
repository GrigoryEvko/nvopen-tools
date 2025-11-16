// Function: sub_16D1EB0
// Address: 0x16d1eb0
//
__int64 __fastcall sub_16D1EB0(__int64 a1)
{
  unsigned __int64 v1; // rax
  unsigned int v2; // r8d
  _WORD *v3; // rdx

  v1 = *(_QWORD *)(a1 + 8);
  v2 = 10;
  if ( v1 <= 1 )
    return v2;
  v3 = *(_WORD **)a1;
  if ( **(_WORD **)a1 == 30768 || *v3 == 22576 )
  {
    v2 = 16;
    *(_QWORD *)a1 = v3 + 1;
    *(_QWORD *)(a1 + 8) = v1 - 2;
    return v2;
  }
  if ( *v3 != 25136 && *v3 != 16944 )
  {
    if ( *v3 == 28464 )
    {
      v2 = 8;
      *(_QWORD *)a1 = v3 + 1;
      *(_QWORD *)(a1 + 8) = v1 - 2;
    }
    else if ( *(_BYTE *)v3 == 48 && (unsigned __int8)(*((_BYTE *)v3 + 1) - 48) <= 9u )
    {
      v2 = 8;
      *(_QWORD *)a1 = (char *)v3 + 1;
      *(_QWORD *)(a1 + 8) = v1 - 1;
    }
    return v2;
  }
  *(_QWORD *)(a1 + 8) = v1 - 2;
  *(_QWORD *)a1 = v3 + 1;
  return 2;
}
