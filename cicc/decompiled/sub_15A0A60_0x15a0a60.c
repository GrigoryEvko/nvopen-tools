// Function: sub_15A0A60
// Address: 0x15a0a60
//
__int64 __fastcall sub_15A0A60(__int64 a1, unsigned int a2)
{
  unsigned int v2; // edx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx

  v2 = *(unsigned __int8 *)(a1 + 16);
  if ( (unsigned __int8)v2 <= 5u )
  {
LABEL_2:
    if ( v2 - 11 <= 1 && (unsigned int)sub_15958F0(a1) > a2 )
      return sub_15A0940(a1, a2);
    return 0;
  }
  if ( v2 > 8 )
  {
    if ( (_BYTE)v2 == 10 )
    {
      if ( (unsigned int)sub_1594180((__int64 *)a1) <= a2 )
        return 0;
      return sub_15A08E0(a1, a2, v5, v6);
    }
    else
    {
      if ( (_BYTE)v2 != 9 )
        goto LABEL_2;
      if ( (unsigned int)sub_15941A0((__int64 *)a1) <= a2 )
        return 0;
      return sub_159A1E0(a1, a2);
    }
  }
  else
  {
    v4 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
    if ( a2 >= (unsigned int)v4 )
      return 0;
    return *(_QWORD *)(a1 + 24 * (a2 - v4));
  }
}
