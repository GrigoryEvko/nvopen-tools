// Function: sub_31D5EF0
// Address: 0x31d5ef0
//
__int64 __fastcall sub_31D5EF0(__int64 a1, __int64 a2)
{
  unsigned int v3; // r12d
  __int64 v5; // rdi
  __int64 v6; // rax

  if ( *(_BYTE *)(a2 + 782) )
    return 1;
  if ( *(_QWORD *)(a1 + 432) != *(_QWORD *)(a1 + 440) )
    return 1;
  v3 = *(unsigned __int8 *)(a1 + 580);
  if ( (_BYTE)v3 )
    return 1;
  v5 = *(_QWORD *)a1;
  if ( (*(_BYTE *)(v5 + 7) & 0x20) == 0 )
    goto LABEL_9;
  if ( sub_B91C10(v5, 37) )
    return 1;
  v5 = *(_QWORD *)a1;
LABEL_9:
  if ( (*(_BYTE *)(v5 + 2) & 8) != 0 )
  {
    v6 = sub_B2E500(v5);
    LOBYTE(v3) = (unsigned int)sub_B2A630(v6) == 0;
  }
  return v3;
}
