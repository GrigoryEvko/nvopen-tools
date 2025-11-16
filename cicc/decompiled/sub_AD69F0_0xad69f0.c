// Function: sub_AD69F0
// Address: 0xad69f0
//
__int64 __fastcall sub_AD69F0(unsigned __int8 *a1, __int64 a2)
{
  __int64 v2; // rdx
  char v3; // al
  __int64 v4; // rcx
  __int64 result; // rax
  __int64 v6; // rax
  unsigned int v7; // r8d
  __int64 *v8; // rax
  __int64 *v9; // rax

  v2 = *a1;
  v3 = *a1;
  if ( (unsigned __int8)v2 <= 8u )
  {
    v4 = *((_QWORD *)a1 + 1);
    goto LABEL_3;
  }
  if ( (unsigned int)v2 <= 0xB )
  {
    v6 = *((_DWORD *)a1 + 1) & 0x7FFFFFF;
    if ( (unsigned int)v6 > (unsigned int)a2 )
      return *(_QWORD *)&a1[32 * ((unsigned int)a2 - v6)];
    return 0;
  }
  if ( (_BYTE)v2 != 14 )
  {
    v4 = *((_QWORD *)a1 + 1);
    if ( (_BYTE)v2 != 17 )
    {
LABEL_3:
      if ( v3 == 18 )
      {
        if ( (unsigned int)a2 < *(_DWORD *)(v4 + 32) )
        {
          v9 = (__int64 *)sub_BD5C60(a1, a2, v2);
          return sub_AC8EA0(v9, (__int64 *)a1 + 3);
        }
      }
      else if ( *(_BYTE *)(v4 + 8) != 18 )
      {
        if ( v3 == 13 )
        {
          if ( (unsigned int)sub_AC3250((__int64)a1) > (unsigned int)a2 )
            return sub_ACB0F0((__int64)a1, a2);
        }
        else if ( (unsigned int)(v2 - 12) > 1 )
        {
          if ( (unsigned int)(v2 - 15) <= 1 && (unsigned int)sub_AC5290((__int64)a1) > (unsigned int)a2 )
            return sub_AD68C0((__int64)a1, a2);
        }
        else if ( (unsigned int)sub_AC3250((__int64)a1) > (unsigned int)a2 )
        {
          return sub_ACABB0((__int64)a1, a2);
        }
      }
      return 0;
    }
    if ( (unsigned int)a2 < *(_DWORD *)(v4 + 32) )
    {
      v8 = (__int64 *)sub_BD5C60(a1, a2, v2);
      return sub_ACCFD0(v8, (__int64)(a1 + 24));
    }
    return 0;
  }
  v7 = sub_AC31F0((__int64)a1);
  result = 0;
  if ( (unsigned int)a2 < v7 )
    return sub_AD6690((__int64)a1, (unsigned int)a2);
  return result;
}
