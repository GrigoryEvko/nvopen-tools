// Function: sub_8CA330
// Address: 0x8ca330
//
__int64 __fastcall sub_8CA330(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax

  if ( qword_4F074B0 != qword_4F60258 )
    return a1;
  result = 0;
  if ( !a1 )
    return result;
  if ( !*qword_4D03FD0 )
    return a1;
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) <= 2u )
  {
    v2 = *(_QWORD *)(a1 + 168);
    if ( v2 )
    {
      v3 = *(_QWORD *)(v2 + 160);
      if ( v3 )
      {
        if ( *(_BYTE *)(v3 + 120) == 8 )
          return a1;
      }
    }
  }
  if ( !unk_4D03FC4 && (!unk_4D03FC0 || (*(_BYTE *)(a1 + 89) & 4) == 0) )
  {
    v4 = *(_QWORD *)(a1 + 32);
    if ( v4 )
      return *(_QWORD *)v4;
    return a1;
  }
  v4 = *(_QWORD *)(a1 + 32);
  if ( v4 )
    return *(_QWORD *)v4;
  sub_8C9400(a1, 6);
  v5 = *(_QWORD *)(a1 + 32);
  if ( v5 )
    return *(_QWORD *)v5;
  else
    return a1;
}
