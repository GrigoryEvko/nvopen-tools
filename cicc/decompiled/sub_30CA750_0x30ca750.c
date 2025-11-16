// Function: sub_30CA750
// Address: 0x30ca750
//
__int64 __fastcall sub_30CA750(__int64 a1)
{
  _BYTE *v1; // rax
  unsigned __int8 v2; // dl
  _BYTE **v3; // rax
  unsigned __int8 v5; // dl
  __int64 *v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rsi

  if ( !sub_B92180(a1) )
    return 1;
  if ( !sub_B92180(a1) )
    return 0;
  v1 = (_BYTE *)sub_B92180(a1);
  if ( *v1 != 16 )
  {
    v2 = *(v1 - 16);
    v3 = (v2 & 2) != 0 ? (_BYTE **)*((_QWORD *)v1 - 4) : (_BYTE **)&v1[-8 * ((v2 >> 2) & 0xF) - 16];
    v1 = *v3;
    if ( !v1 )
      return sub_2C75D80((__int64)byte_3F871B3, 0);
  }
  v5 = *(v1 - 16);
  if ( (v5 & 2) != 0 )
    v6 = (__int64 *)*((_QWORD *)v1 - 4);
  else
    v6 = (__int64 *)&v1[-8 * ((v5 >> 2) & 0xF) - 16];
  v7 = *v6;
  if ( *v6 )
  {
    v7 = sub_B91420(v7);
    v9 = v8;
  }
  else
  {
    v9 = 0;
  }
  return sub_2C75D80(v7, v9);
}
