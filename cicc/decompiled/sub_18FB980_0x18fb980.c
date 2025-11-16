// Function: sub_18FB980
// Address: 0x18fb980
//
__int64 __fastcall sub_18FB980(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  unsigned int v4; // eax
  int v5; // ecx
  int v6; // r8d
  int v7; // eax
  int v8; // r13d
  __int64 v9; // [rsp+18h] [rbp-58h] BYREF
  __int64 v10; // [rsp+20h] [rbp-50h] BYREF
  __int64 v11; // [rsp+28h] [rbp-48h] BYREF
  __int64 v12[8]; // [rsp+30h] [rbp-40h] BYREF

  LOBYTE(v2) = a2 == -8 || a1 == -8 || a1 == -16 || a2 == -16;
  if ( (_BYTE)v2 )
  {
    LOBYTE(v2) = a1 == a2;
    return v2;
  }
  if ( *(_BYTE *)(a2 + 16) != *(_BYTE *)(a1 + 16) )
    return v2;
  LOBYTE(v4) = sub_15F40E0(a1, a2);
  if ( (_BYTE)v4 )
    return v4;
  v5 = *(unsigned __int8 *)(a1 + 16);
  if ( (unsigned int)(v5 - 35) <= 0x11 )
  {
    if ( ((1LL << ((unsigned __int8)v5 - 24)) & 0x1C019800) != 0 && *(_QWORD *)(a2 - 24) == *(_QWORD *)(a1 - 48) )
      LOBYTE(v2) = *(_QWORD *)(a2 - 48) == *(_QWORD *)(a1 - 24);
    return v2;
  }
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 16) - 75) > 1u )
  {
    v8 = sub_14B2890(a1, &v9, &v10, 0, 0);
    if ( (unsigned int)(v8 - 1) <= 3 )
    {
      if ( v8 == (unsigned int)sub_14B2890(a2, &v11, v12, 0, 0)
        && (v9 == v11 && v10 == v12[0] || v9 == v12[0] && v11 == v10) )
      {
        return 1;
      }
    }
    else
    {
      if ( (unsigned int)(v8 - 7) > 1 )
        return v2;
      if ( v8 == (unsigned int)sub_14B2890(a2, &v11, v12, 0, 0) && v9 == v11 && v10 == v12[0] )
        return 1;
    }
    return 0;
  }
  if ( *(_QWORD *)(a2 - 24) == *(_QWORD *)(a1 - 48) && *(_QWORD *)(a2 - 48) == *(_QWORD *)(a1 - 24) )
  {
    v6 = sub_15FF5D0(*(_WORD *)(a1 + 18) & 0x7FFF);
    v7 = *(unsigned __int16 *)(a2 + 18);
    BYTE1(v7) &= ~0x80u;
    LOBYTE(v2) = v7 == v6;
  }
  return v2;
}
