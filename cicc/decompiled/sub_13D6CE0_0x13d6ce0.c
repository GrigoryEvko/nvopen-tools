// Function: sub_13D6CE0
// Address: 0x13d6ce0
//
__int64 __fastcall sub_13D6CE0(_QWORD *a1, __int64 a2, char a3, _QWORD *a4)
{
  unsigned __int8 v5; // al
  __int64 v6; // r13
  char v8; // al
  double v9; // xmm0_8
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rcx
  double v13[5]; // [rsp+8h] [rbp-28h] BYREF

  v5 = *((_BYTE *)a1 + 16);
  if ( v5 <= 0x10u )
  {
    if ( *(_BYTE *)(a2 + 16) <= 0x10u )
    {
      v6 = sub_14D6F90(19, a1, a2, *a4);
      if ( v6 )
        return v6;
      v5 = *((_BYTE *)a1 + 16);
    }
    if ( v5 == 9 )
      goto LABEL_8;
  }
  if ( *(_BYTE *)(a2 + 16) == 9 )
LABEL_8:
    v6 = sub_15A11D0(*a1, 0, 0);
  else
    v6 = sub_13CDA40((unsigned __int8 *)a1, (_QWORD *)a2);
  if ( v6 )
    return v6;
  v13[0] = 1.0;
  if ( (unsigned __int8)sub_13D6AF0(v13, a2) )
    return (__int64)a1;
  if ( (a3 & 2) == 0 )
    return v6;
  if ( (a3 & 8) == 0 || !(unsigned __int8)sub_13CC1F0((__int64)a1) )
  {
    if ( a1 == (_QWORD *)a2 )
    {
      v9 = 1.0;
    }
    else
    {
      if ( (a3 & 1) != 0 )
      {
        v8 = *((_BYTE *)a1 + 16);
        if ( v8 == 40 )
        {
          v10 = *(a1 - 6);
          v11 = *(a1 - 3);
          if ( v10 && a2 == v11 )
            return *(a1 - 6);
          if ( v11 && v10 == a2 )
            return v11;
        }
        else if ( v8 == 5 && *((_WORD *)a1 + 9) == 16 )
        {
          v12 = a1[-3 * (*((_DWORD *)a1 + 5) & 0xFFFFFFF)];
          v11 = a1[3 * (1LL - (*((_DWORD *)a1 + 5) & 0xFFFFFFF))];
          if ( v12 && v11 == a2 )
            return a1[-3 * (*((_DWORD *)a1 + 5) & 0xFFFFFFF)];
          if ( v11 && v12 == a2 )
            return v11;
        }
      }
      if ( (!(unsigned __int8)sub_15FB6D0(a1, 1) || a2 != sub_15FB7A0(a1))
        && (!(unsigned __int8)sub_15FB6D0(a2, 1) || a1 != (_QWORD *)sub_15FB7A0(a2)) )
      {
        return v6;
      }
      v9 = -1.0;
    }
    return sub_15A10B0(*a1, v9);
  }
  return sub_15A06D0(*a1);
}
