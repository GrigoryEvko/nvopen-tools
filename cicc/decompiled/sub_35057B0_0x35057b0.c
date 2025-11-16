// Function: sub_35057B0
// Address: 0x35057b0
//
__int64 __fastcall sub_35057B0(_QWORD *a1, unsigned __int8 *a2, __int64 a3)
{
  __int64 v4; // r12
  unsigned __int8 *v5; // rax
  unsigned __int8 v6; // dl
  unsigned __int8 v7; // al
  __int64 v8; // rdx
  __int64 v10; // rcx
  __int64 v11; // rax

  if ( !a3 )
    return sub_35058C0(a1, a2);
  v4 = a3;
  while ( 1 )
  {
    v5 = sub_AF34D0(a2);
    v6 = *(v5 - 16);
    if ( (v6 & 2) != 0 )
    {
      if ( *(_DWORD *)(*(_QWORD *)(*((_QWORD *)v5 - 4) + 40LL) + 32LL) )
        goto LABEL_10;
    }
    else if ( *(_DWORD *)(*(_QWORD *)&v5[-8 * ((v6 >> 2) & 0xF) + 24] + 32LL) )
    {
LABEL_10:
      sub_3504CA0((__int64)a1, a2);
      return sub_3505300(a1, a2, v4);
    }
    v7 = *(_BYTE *)(v4 - 16);
    if ( (v7 & 2) == 0 )
      break;
    v8 = *(_QWORD *)(v4 - 32);
    if ( *(_DWORD *)(v4 - 24) != 2 )
    {
      a2 = *(unsigned __int8 **)v8;
      return sub_35058C0(a1, a2);
    }
    v4 = *(_QWORD *)(v8 + 8);
LABEL_13:
    a2 = *(unsigned __int8 **)v8;
    if ( !v4 )
      return sub_35058C0(a1, a2);
  }
  v10 = v4 - 16;
  v11 = 8LL * ((v7 >> 2) & 0xF);
  if ( ((*(_WORD *)(v4 - 16) >> 6) & 0xF) == 2 )
  {
    v4 = *(_QWORD *)(v10 - v11 + 8);
    v8 = v10 - v11;
    goto LABEL_13;
  }
  a2 = *(unsigned __int8 **)(v10 - v11);
  return sub_35058C0(a1, a2);
}
