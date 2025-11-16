// Function: sub_2FD5CE0
// Address: 0x2fd5ce0
//
__int64 __fastcall sub_2FD5CE0(int a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v6; // rdx

  if ( a1 < 0 )
    v3 = *(_QWORD *)(*(_QWORD *)(a3 + 56) + 16LL * (a1 & 0x7FFFFFFF) + 8);
  else
    v3 = *(_QWORD *)(*(_QWORD *)(a3 + 304) + 8LL * (unsigned int)a1);
  if ( !v3 )
    return 0;
  if ( (*(_BYTE *)(v3 + 3) & 0x10) == 0 )
  {
LABEL_5:
    v4 = *(_QWORD *)(v3 + 16);
    if ( (unsigned __int16)(*(_WORD *)(v4 + 68) - 14) <= 1u )
      goto LABEL_8;
LABEL_15:
    if ( a2 != *(_QWORD *)(v4 + 24) )
      return 1;
LABEL_8:
    while ( 1 )
    {
      v3 = *(_QWORD *)(v3 + 32);
      if ( !v3 )
        return 0;
      if ( (*(_BYTE *)(v3 + 3) & 0x10) == 0 )
      {
        v6 = *(_QWORD *)(v3 + 16);
        if ( v6 != v4 )
        {
          v4 = *(_QWORD *)(v3 + 16);
          if ( (unsigned __int16)(*(_WORD *)(v6 + 68) - 14) > 1u )
            goto LABEL_15;
        }
      }
    }
  }
  while ( 1 )
  {
    v3 = *(_QWORD *)(v3 + 32);
    if ( !v3 )
      return 0;
    if ( (*(_BYTE *)(v3 + 3) & 0x10) == 0 )
      goto LABEL_5;
  }
}
