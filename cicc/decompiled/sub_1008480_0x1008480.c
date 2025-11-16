// Function: sub_1008480
// Address: 0x1008480
//
__int64 __fastcall sub_1008480(__int64 *a1, unsigned __int8 *a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 v8; // r13
  _BYTE *v9; // rdi
  __int64 v10; // rbx
  int v11; // eax
  unsigned __int8 *v13; // rbx
  __int64 v14; // rdx
  __int64 v15; // rdx

  if ( *a2 <= 0x1Cu )
    return 0;
  v3 = *((_QWORD *)a2 + 1);
  if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 <= 1 )
    v3 = **(_QWORD **)(v3 + 16);
  if ( !sub_BCAC40(v3, 1) )
    return 0;
  LODWORD(v7) = *a2;
  if ( (_BYTE)v7 == 58 )
  {
    if ( (a2[7] & 0x40) != 0 )
    {
      v13 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
    }
    else
    {
      v7 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
      v13 = &a2[-v7];
    }
    v14 = *a1;
    LOBYTE(v7) = *(_QWORD *)v13 == *a1;
    LOBYTE(v14) = *((_QWORD *)v13 + 4) == *a1;
    return (unsigned int)v14 | (unsigned int)v7;
  }
  else
  {
    if ( (_BYTE)v7 != 86 )
      return 0;
    v8 = *((_QWORD *)a2 - 12);
    if ( *((_QWORD *)a2 + 1) != *(_QWORD *)(v8 + 8) )
      return 0;
    v9 = (_BYTE *)*((_QWORD *)a2 - 8);
    if ( *v9 > 0x15u )
      return 0;
    v10 = *((_QWORD *)a2 - 4);
    if ( !sub_AD7A80(v9, 1, v4, v5, v6) )
      return 0;
    v15 = *a1;
    LOBYTE(v11) = v10 == *a1;
    LOBYTE(v15) = v8 == *a1;
    return (unsigned int)v15 | v11;
  }
}
