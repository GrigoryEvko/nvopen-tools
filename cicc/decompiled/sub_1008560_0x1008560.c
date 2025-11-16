// Function: sub_1008560
// Address: 0x1008560
//
__int64 __fastcall sub_1008560(__int64 *a1, unsigned __int8 *a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // r13
  _BYTE *v6; // rdi
  __int64 v7; // rbx
  int v8; // eax
  unsigned __int8 *v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // rdx

  if ( *a2 <= 0x1Cu )
    return 0;
  v3 = *((_QWORD *)a2 + 1);
  if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 <= 1 )
    v3 = **(_QWORD **)(v3 + 16);
  if ( !sub_BCAC40(v3, 1) )
    return 0;
  LODWORD(v4) = *a2;
  if ( (_BYTE)v4 == 57 )
  {
    if ( (a2[7] & 0x40) != 0 )
    {
      v10 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
    }
    else
    {
      v4 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
      v10 = &a2[-v4];
    }
    v11 = *a1;
    LOBYTE(v4) = *(_QWORD *)v10 == *a1;
    LOBYTE(v11) = *((_QWORD *)v10 + 4) == *a1;
    return (unsigned int)v11 | (unsigned int)v4;
  }
  else
  {
    if ( (_BYTE)v4 != 86 )
      return 0;
    v5 = *((_QWORD *)a2 - 12);
    if ( *((_QWORD *)a2 + 1) != *(_QWORD *)(v5 + 8) )
      return 0;
    v6 = (_BYTE *)*((_QWORD *)a2 - 4);
    if ( *v6 > 0x15u )
      return 0;
    v7 = *((_QWORD *)a2 - 8);
    if ( !sub_AC30F0((__int64)v6) )
      return 0;
    v12 = *a1;
    LOBYTE(v8) = v7 == *a1;
    LOBYTE(v12) = v5 == *a1;
    return (unsigned int)v12 | v8;
  }
}
