// Function: sub_8DF8D0
// Address: 0x8df8d0
//
__int64 __fastcall sub_8DF8D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  __int64 v6; // r12
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 result; // rax
  char v10; // al

  v5 = a2;
  v6 = a1;
  if ( *(_BYTE *)(a1 + 140) != 12 )
    goto LABEL_5;
  do
    v6 = *(_QWORD *)(v6 + 160);
  while ( *(_BYTE *)(v6 + 140) == 12 );
  if ( *(_BYTE *)(a2 + 140) == 12 )
  {
    do
    {
      v5 = *(_QWORD *)(v5 + 160);
LABEL_5:
      ;
    }
    while ( *(_BYTE *)(v5 + 140) == 12 );
  }
  if ( v6 == v5
    || (unsigned int)sub_8D97D0(v6, v5, 0, a4, a5)
    || (unsigned __int8)(*(_BYTE *)(v6 + 140) - 9) <= 2u
    && (unsigned __int8)(*(_BYTE *)(v5 + 140) - 9) <= 2u
    && sub_8D5CE0(v5, v6) )
  {
    return 1;
  }
  if ( !dword_4F077BC )
  {
    LOBYTE(result) = (unsigned int)sub_8DAFE0(v6, v5) != 0;
    return (unsigned __int8)result;
  }
  if ( qword_4F077A8 <= 0x1869Fu )
  {
    LOBYTE(result) = (unsigned int)sub_8DED30(v6, v5, 5, v7, v8) != 0;
    return (unsigned __int8)result;
  }
  result = sub_8DAFE0(v6, v5);
  if ( (_DWORD)result )
  {
    while ( 1 )
    {
      v10 = *(_BYTE *)(v6 + 140);
      if ( v10 != 12 )
        break;
      v6 = *(_QWORD *)(v6 + 160);
    }
    if ( v10 == 6 )
      return !sub_8D3A70(*(_QWORD *)(v6 + 160));
    return 1;
  }
  return result;
}
