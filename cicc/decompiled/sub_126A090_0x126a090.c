// Function: sub_126A090
// Address: 0x126a090
//
__int64 __fastcall sub_126A090(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r12
  __int64 result; // rax
  __int64 i; // rdi
  bool v8; // zf

  sub_15E5440(a2, a3);
  v5 = *(_QWORD *)(a4 + 120);
  result = sub_127BF90(v5);
  if ( !(_BYTE)result )
  {
    if ( sub_8D3410(v5) )
      v5 = sub_8D40F0(v5);
    for ( i = v5; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v8 = !sub_8D3A70(i);
    result = *(unsigned __int8 *)(v5 + 140);
    if ( !v8 )
    {
      if ( (_BYTE)result == 12 )
      {
        result = v5;
        do
          result = *(_QWORD *)(result + 160);
        while ( *(_BYTE *)(result + 140) == 12 );
        if ( (*(_BYTE *)(result + 176) & 8) != 0 )
          return result;
LABEL_12:
        result = sub_8D4C10(v5, dword_4F077C4 != 2);
        if ( (result & 1) != 0 )
          *(_BYTE *)(a2 + 80) |= 1u;
        return result;
      }
      if ( (*(_BYTE *)(v5 + 176) & 8) != 0 )
        return result;
    }
    result = (unsigned int)result & 0xFFFFFFFB;
    if ( (_BYTE)result != 8 )
      return result;
    goto LABEL_12;
  }
  return result;
}
