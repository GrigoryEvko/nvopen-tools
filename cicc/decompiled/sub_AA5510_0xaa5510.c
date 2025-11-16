// Function: sub_AA5510
// Address: 0xaa5510
//
__int64 __fastcall sub_AA5510(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // rcx

  v1 = *(_QWORD *)(a1 + 16);
  do
  {
    if ( !v1 )
      return 0;
    v2 = *(_QWORD *)(v1 + 24);
    v1 = *(_QWORD *)(v1 + 8);
  }
  while ( (unsigned __int8)(*(_BYTE *)v2 - 30) > 0xAu );
  v3 = *(_QWORD *)(v2 + 40);
  if ( v1 )
  {
    while ( 1 )
    {
      v4 = *(_QWORD *)(v1 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v4 - 30) <= 0xAu )
        break;
      v1 = *(_QWORD *)(v1 + 8);
      if ( !v1 )
        return v3;
    }
LABEL_8:
    if ( v3 != *(_QWORD *)(v4 + 40) )
      return 0;
    while ( 1 )
    {
      v1 = *(_QWORD *)(v1 + 8);
      if ( !v1 )
        break;
      v4 = *(_QWORD *)(v1 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v4 - 30) <= 0xAu )
        goto LABEL_8;
    }
  }
  return v3;
}
