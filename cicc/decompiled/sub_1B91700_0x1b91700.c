// Function: sub_1B91700
// Address: 0x1b91700
//
__int64 __fastcall sub_1B91700(__int64 a1, __int64 a2)
{
  _DWORD *v4; // rcx
  _DWORD *v5; // rsi
  int v6; // edx
  _DWORD *v7; // rax

  do
  {
LABEL_1:
    v4 = *(_DWORD **)(a1 + 24);
    v5 = &v4[4 * *(unsigned int *)(a1 + 40)];
  }
  while ( v4 == v5 );
  while ( 1 )
  {
    v6 = *v4;
    v7 = v4;
    if ( (unsigned int)(*v4 + 0x7FFFFFFF) <= 0xFFFFFFFD )
      break;
    v4 += 4;
    if ( v5 == v4 )
      goto LABEL_1;
  }
  while ( 1 )
  {
    if ( v5 == v7 )
      goto LABEL_1;
    if ( *((_QWORD *)v7 + 1) == a2 )
      return (unsigned int)(v6 - *(_DWORD *)(a1 + 48));
    v7 += 4;
    if ( v7 == v5 )
      goto LABEL_1;
    while ( 1 )
    {
      v6 = *v7;
      if ( (unsigned int)(*v7 + 0x7FFFFFFF) <= 0xFFFFFFFD )
        break;
      v7 += 4;
    }
  }
}
