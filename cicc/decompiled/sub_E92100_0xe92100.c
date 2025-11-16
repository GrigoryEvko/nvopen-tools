// Function: sub_E92100
// Address: 0xe92100
//
__int64 __fastcall sub_E92100(__int64 a1, unsigned int a2)
{
  unsigned __int16 *v2; // rdx
  __int64 v3; // rax
  unsigned __int16 v4; // dx
  unsigned int v5; // r8d

  v2 = (unsigned __int16 *)(*(_QWORD *)(a1 + 48) + 4LL * a2);
  v3 = *v2;
  v4 = v2[1];
  while ( (_WORD)v3 )
  {
    v5 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 8) + 24 * v3 + 23);
    v3 = v4;
    v4 = 0;
    if ( (_BYTE)v5 )
      return v5;
  }
  return 0;
}
