// Function: sub_30742E0
// Address: 0x30742e0
//
char __fastcall sub_30742E0(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // r12
  __int64 v3; // rax

  v2 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 == 22 )
  {
    LODWORD(v3) = sub_CE9220(*(_QWORD *)(a2 + 24)) ^ 1;
    return v3;
  }
  if ( v2 == 60 )
    goto LABEL_14;
  LOBYTE(v3) = 0;
  if ( v2 <= 0x1Cu )
    return v3;
  if ( v2 != 61 )
  {
    LOBYTE(v3) = sub_B46500((unsigned __int8 *)a2);
    if ( (_BYTE)v3 || v2 != 85 )
      return v3;
    v3 = *(_QWORD *)(a2 - 32);
    if ( *(_BYTE *)v3 == 25 )
    {
      LOBYTE(v3) = *(_BYTE *)(v3 + 96);
      return v3;
    }
LABEL_14:
    LOBYTE(v3) = 1;
    return v3;
  }
  v3 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
  if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 <= 1 )
    v3 = **(_QWORD **)(v3 + 16);
  LOBYTE(v3) = *(_DWORD *)(v3 + 8) >> 8 == 5;
  return v3;
}
