// Function: sub_134BD00
// Address: 0x134bd00
//
__int64 __fastcall sub_134BD00(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 result; // rax
  __int64 v4; // rax

  v2 = a2;
  *(_BYTE *)(a2 + 35) = 0;
  sub_134B960(a1, a2);
  if ( *(_BYTE *)(a2 + 17) )
    sub_134B7F0((__int64)a1, a2);
  if ( *(_BYTE *)(a2 + 19) )
    sub_134B5B0((__int64)a1, a2);
  result = *(unsigned __int8 *)(a2 + 32);
  if ( *(_BYTE *)(a2 + 20) )
  {
    if ( !(_BYTE)result )
    {
      *(_BYTE *)(a2 + 32) = 1;
      *(_QWORD *)(a2 + 80) = a2;
      *(_QWORD *)(a2 + 88) = a2;
      result = a1[659];
      if ( result )
      {
        *(_QWORD *)(a2 + 80) = *(_QWORD *)(result + 88);
        *(_QWORD *)(a1[659] + 88LL) = a2;
        *(_QWORD *)(a2 + 88) = *(_QWORD *)(*(_QWORD *)(a2 + 88) + 80LL);
        *(_QWORD *)(*(_QWORD *)(a1[659] + 88LL) + 80LL) = a1[659];
        result = *(_QWORD *)(a2 + 88);
        *(_QWORD *)(result + 80) = a2;
        v2 = *(_QWORD *)(a2 + 80);
      }
      a1[659] = v2;
    }
  }
  else if ( (_BYTE)result )
  {
    *(_BYTE *)(a2 + 32) = 0;
    if ( a2 == a1[659] )
    {
      result = *(_QWORD *)(a2 + 80);
      if ( a2 == result )
      {
        a1[659] = 0;
        return result;
      }
      a1[659] = result;
    }
    *(_QWORD *)(*(_QWORD *)(a2 + 88) + 80LL) = *(_QWORD *)(*(_QWORD *)(a2 + 80) + 88LL);
    v4 = *(_QWORD *)(a2 + 88);
    *(_QWORD *)(*(_QWORD *)(a2 + 80) + 88LL) = v4;
    *(_QWORD *)(a2 + 88) = *(_QWORD *)(v4 + 80);
    *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 80) + 88LL) + 80LL) = *(_QWORD *)(a2 + 80);
    result = *(_QWORD *)(a2 + 88);
    *(_QWORD *)(result + 80) = a2;
  }
  return result;
}
