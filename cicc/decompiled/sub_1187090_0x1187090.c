// Function: sub_1187090
// Address: 0x1187090
//
__int64 __fastcall sub_1187090(_QWORD **a1, __int64 a2)
{
  __int64 v3; // rdi
  unsigned int v4; // eax
  unsigned int v5; // r12d
  __int64 v6; // r14
  _BYTE *v8; // rdi
  unsigned __int8 *v9; // r15
  __int64 *v10; // rbx
  _BYTE *v11; // rbx

  if ( !a2 )
    return 0;
  v3 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 <= 1 )
    v3 = **(_QWORD **)(v3 + 16);
  LOBYTE(v4) = sub_BCAC40(v3, 1);
  v5 = v4;
  if ( !(_BYTE)v4 )
    return 0;
  if ( *(_BYTE *)a2 == 57 )
  {
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v10 = *(__int64 **)(a2 - 8);
    else
      v10 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    v6 = *v10;
    v11 = (_BYTE *)v10[4];
    if ( *(_BYTE *)v6 == 59
      && ((unsigned __int8)sub_995B10(a1, *(_QWORD *)(v6 - 64)) && sub_1181310(a1 + 1, *(_QWORD *)(v6 - 32))
       || (unsigned __int8)sub_995B10(a1, *(_QWORD *)(v6 - 32)) && sub_1181310(a1 + 1, *(_QWORD *)(v6 - 64)))
      && (unsigned __int8)sub_1186D70(a1 + 3, (__int64)v11) )
    {
      return v5;
    }
    if ( *v11 == 59
      && ((unsigned __int8)sub_995B10(a1, *((_QWORD *)v11 - 8)) && sub_1181310(a1 + 1, *((_QWORD *)v11 - 4))
       || (unsigned __int8)sub_995B10(a1, *((_QWORD *)v11 - 4)) && sub_1181310(a1 + 1, *((_QWORD *)v11 - 8))) )
    {
      return sub_1186D70(a1 + 3, v6);
    }
    return 0;
  }
  if ( *(_BYTE *)a2 != 86 )
    return 0;
  v6 = *(_QWORD *)(a2 - 96);
  if ( *(_QWORD *)(a2 + 8) != *(_QWORD *)(v6 + 8) )
    return 0;
  v8 = *(_BYTE **)(a2 - 32);
  if ( *v8 > 0x15u )
    return 0;
  v9 = *(unsigned __int8 **)(a2 - 64);
  if ( !sub_AC30F0((__int64)v8) )
    return 0;
  if ( sub_1186E90(a1, 30, (unsigned __int8 *)v6) && (unsigned __int8)sub_1186D70(a1 + 3, (__int64)v9) )
    return v5;
  if ( !sub_1186E90(a1, 30, v9) )
    return 0;
  return sub_1186D70(a1 + 3, v6);
}
