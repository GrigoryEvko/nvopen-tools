// Function: sub_3952EB0
// Address: 0x3952eb0
//
__int64 __fastcall sub_3952EB0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r13
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rdx
  __int64 v6; // rdx
  __int64 v7; // rbx
  unsigned __int64 v8; // rax

  if ( *(_BYTE *)(a1 + 16) <= 0x10u )
    return 0;
  result = 0;
  v3 = *(_QWORD *)a1;
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) )
  {
    if ( sub_1642F90(*(_QWORD *)a1, 1) )
      return 0x100000000LL;
    v4 = *(unsigned __int8 *)(v3 + 8);
    if ( (_BYTE)v4 == 13 )
    {
      LODWORD(v5) = *(_DWORD *)sub_15A9930(a2, v3);
    }
    else if ( *(_BYTE *)(a1 + 16) == 77 && (_BYTE)v4 == 16 )
    {
      v7 = *(_QWORD *)(v3 + 32);
      v8 = (unsigned __int64)(sub_127FA20(a2, *(_QWORD *)(v3 + 24)) + 7) >> 3;
      if ( (int)v8 > 4 )
        LODWORD(v8) = 4;
      LODWORD(v5) = v7 * v8;
    }
    else
    {
      if ( (unsigned __int8)v4 > 0xFu || (v6 = 35454, !_bittest64(&v6, v4)) )
      {
        if ( (unsigned int)(v4 - 13) > 1 && (_DWORD)v4 != 16 || !sub_16435F0(v3, 0) )
          return 1;
      }
      v5 = (unsigned __int64)(sub_127FA20(a2, v3) + 7) >> 3;
    }
    result = (unsigned int)((int)v5 / 4);
    if ( (int)v5 > 3 )
      return result;
    return 1;
  }
  return result;
}
