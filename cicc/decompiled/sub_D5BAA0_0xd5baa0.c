// Function: sub_D5BAA0
// Address: 0xd5baa0
//
__int64 __fastcall sub_D5BAA0(unsigned __int8 *a1)
{
  int v1; // eax
  unsigned __int64 v2; // rax
  __int64 result; // rax
  __int64 v4; // rax
  __int64 v5; // rdx

  v1 = *a1;
  if ( (unsigned __int8)v1 <= 0x1Cu )
    return 0;
  if ( (_BYTE)v1 == 85 )
  {
    v4 = *((_QWORD *)a1 - 4);
    if ( v4 && !*(_BYTE *)v4 && *(_QWORD *)(v4 + 24) == *((_QWORD *)a1 + 10) && (*(_BYTE *)(v4 + 33) & 0x20) != 0 )
      return 0;
  }
  else
  {
    v2 = (unsigned int)(v1 - 34);
    if ( (unsigned __int8)v2 > 0x33u )
      return 0;
    v5 = 0x8000000000041LL;
    if ( !_bittest64(&v5, v2) )
      return 0;
  }
  if ( ((unsigned __int8)sub_A73ED0((_QWORD *)a1 + 9, 23) || (unsigned __int8)sub_B49560((__int64)a1, 23))
    && !(unsigned __int8)sub_A73ED0((_QWORD *)a1 + 9, 4)
    && !(unsigned __int8)sub_B49560((__int64)a1, 4) )
  {
    return 0;
  }
  result = *((_QWORD *)a1 - 4);
  if ( !result || *(_BYTE *)result || *(_QWORD *)(result + 24) != *((_QWORD *)a1 + 10) )
    return 0;
  return result;
}
