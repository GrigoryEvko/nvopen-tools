// Function: sub_828CC0
// Address: 0x828cc0
//
__int64 __fastcall sub_828CC0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v4; // rsi
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 i; // rax

  v2 = a2;
  if ( *a1 == a2 || (unsigned int)sub_8DED30(a2, *a1, 1) )
    return 3;
  v4 = *a1;
  if ( (unsigned int)sub_8DED40(v2, *a1) )
    return 4;
  if ( dword_4D04964
    || !(unsigned int)sub_8D2E30(v2)
    || !(unsigned int)sub_8D2930(*a1)
    || !(unsigned int)sub_6E97C0((__int64)a1, v4, v5, v6) )
  {
    return 5;
  }
  while ( *(_BYTE *)(v2 + 140) == 12 )
    v2 = *(_QWORD *)(v2 + 160);
  for ( i = *a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( *(_QWORD *)(v2 + 128) == *(_QWORD *)(i + 128) )
    return 4;
  else
    return 5;
}
