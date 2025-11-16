// Function: sub_31F58C0
// Address: 0x31f58c0
//
__int64 __fastcall sub_31F58C0(__int64 a1)
{
  __int64 v2; // r12
  unsigned __int8 v3; // al
  __int64 v4; // rdi
  __int64 v5; // rdx
  unsigned int v6; // r14d
  unsigned __int8 *v7; // r12
  unsigned int v9; // eax

  v2 = a1 - 16;
  v3 = *(_BYTE *)(a1 - 16);
  if ( (v3 & 2) != 0 )
  {
    v4 = *(_QWORD *)(*(_QWORD *)(a1 - 32) + 56LL);
    if ( !v4 )
    {
      v6 = 0;
      goto LABEL_15;
    }
  }
  else
  {
    v4 = *(_QWORD *)(v2 - 8LL * ((v3 >> 2) & 0xF) + 56);
    if ( !v4 )
    {
      v6 = 0;
      goto LABEL_6;
    }
  }
  sub_B91420(v4);
  v3 = *(_BYTE *)(a1 - 16);
  if ( v5 )
    v6 = 512;
  else
    v6 = 0;
  if ( (*(_BYTE *)(a1 - 16) & 2) == 0 )
  {
LABEL_6:
    v7 = *(unsigned __int8 **)(v2 - 8LL * ((v3 >> 2) & 0xF) + 8);
    if ( v7 )
      goto LABEL_7;
LABEL_16:
    sub_AF18C0(a1);
    return v6;
  }
LABEL_15:
  v7 = *(unsigned __int8 **)(*(_QWORD *)(a1 - 32) + 8LL);
  if ( !v7 )
    goto LABEL_16;
LABEL_7:
  if ( *v7 != 14 )
  {
    if ( (unsigned __int16)sub_AF18C0(a1) != 4 )
      goto LABEL_11;
    goto LABEL_20;
  }
  v6 |= 8u;
  if ( (unsigned __int16)sub_AF18C0(a1) == 4 )
  {
LABEL_20:
    v9 = v6;
    if ( *v7 == 18 )
    {
      BYTE1(v9) = BYTE1(v6) | 1;
      return v9;
    }
    return v6;
  }
LABEL_11:
  while ( *v7 != 18 )
  {
    v7 = (unsigned __int8 *)sub_AF2660(v7);
    if ( !v7 )
      return v6;
  }
  LOWORD(v6) = v6 | 0x100;
  return v6;
}
