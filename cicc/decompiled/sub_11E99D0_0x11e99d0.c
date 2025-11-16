// Function: sub_11E99D0
// Address: 0x11e99d0
//
__int64 __fastcall sub_11E99D0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdi
  unsigned int v5; // r13d
  __int64 *v6; // rax
  __int64 v7; // rdx
  _BYTE *v8; // rax

  if ( !(unsigned __int8)sub_A73ED0((_QWORD *)(a2 + 72), 5) && !(unsigned __int8)sub_B49560(a2, 5) )
  {
    v3 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    if ( *(_BYTE *)v3 == 17 )
    {
      v4 = v3 + 24;
    }
    else
    {
      v7 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v3 + 8) + 8LL) - 17;
      if ( (unsigned int)v7 > 1 )
        return 0;
      if ( *(_BYTE *)v3 > 0x15u )
        return 0;
      v8 = sub_AD7630(v3, 0, v7);
      if ( !v8 || *v8 != 17 )
        return 0;
      v4 = (__int64)(v8 + 24);
    }
    v5 = *(_DWORD *)(v4 + 8);
    if ( v5 <= 0x40 )
    {
      if ( !*(_QWORD *)v4 )
        return 0;
      goto LABEL_8;
    }
    if ( v5 != (unsigned int)sub_C444A0(v4) )
    {
LABEL_8:
      v6 = (__int64 *)sub_BD5C60(a2);
      *(_QWORD *)(a2 + 72) = sub_A7A090((__int64 *)(a2 + 72), v6, -1, 5);
    }
  }
  return 0;
}
