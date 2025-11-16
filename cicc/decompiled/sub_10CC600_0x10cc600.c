// Function: sub_10CC600
// Address: 0x10cc600
//
__int64 __fastcall sub_10CC600(__int64 a1, int a2, unsigned __int8 *a3)
{
  __int64 v5; // rdi
  __int64 v6; // rdx
  _BYTE *v7; // rax

  if ( a2 + 29 != *a3 )
    return 0;
  if ( *(_QWORD *)a1 != *((_QWORD *)a3 - 8) )
    return 0;
  v5 = *((_QWORD *)a3 - 4);
  if ( *(_BYTE *)v5 == 17 )
  {
    **(_QWORD **)(a1 + 8) = v5 + 24;
    return 1;
  }
  else
  {
    v6 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v5 + 8) + 8LL) - 17;
    if ( (unsigned int)v6 > 1 )
      return 0;
    if ( *(_BYTE *)v5 > 0x15u )
      return 0;
    v7 = sub_AD7630(v5, *(unsigned __int8 *)(a1 + 16), v6);
    if ( !v7 || *v7 != 17 )
      return 0;
    **(_QWORD **)(a1 + 8) = v7 + 24;
    return 1;
  }
}
