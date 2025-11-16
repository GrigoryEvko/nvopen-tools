// Function: sub_2B0E4E0
// Address: 0x2b0e4e0
//
bool __fastcall sub_2B0E4E0(__int64 *a1)
{
  __int64 v1; // rsi
  _BYTE *v2; // r8
  bool result; // al
  _BYTE **v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rdx
  bool v7; // cl

  v1 = *a1;
  v2 = *(_BYTE **)(*a1 + 416);
  if ( *(_DWORD *)(*a1 + 104) != 3 )
    goto LABEL_2;
  if ( v2 && *(_QWORD *)(v1 + 424) && *v2 == 90 )
    return *v2 == 84;
  v4 = *(_BYTE ***)v1;
  v5 = *(_QWORD *)v1 + 8LL * *(unsigned int *)(v1 + 8);
  if ( *(_QWORD *)v1 == v5 )
    return 1;
  v6 = 0;
  do
  {
    v7 = **v4++ == 90;
    v6 += v7;
  }
  while ( (_BYTE **)v5 != v4 );
  result = 1;
  if ( v6 > 4 )
  {
LABEL_2:
    result = 0;
    if ( v2 && *(_QWORD *)(v1 + 424) )
      return *v2 == 84;
  }
  return result;
}
