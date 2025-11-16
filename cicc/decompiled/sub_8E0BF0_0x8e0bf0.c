// Function: sub_8E0BF0
// Address: 0x8e0bf0
//
__int64 __fastcall sub_8E0BF0(__int64 a1, __int64 a2, __int64 a3, int a4, int *a5)
{
  __int64 v7; // r13
  __int64 v8; // r12
  __int64 result; // rax
  __int64 v10; // rcx
  __int64 v11; // r8

  v7 = a2;
  v8 = a1;
  *a5 = 0;
  if ( !sub_8D2310(a1) || !sub_8D2310(a2) )
    return sub_8DF7B0(a2, a1, a5, 0, 0);
  while ( *(_BYTE *)(v8 + 140) == 12 )
    v8 = *(_QWORD *)(v8 + 160);
  if ( *(_BYTE *)(a2 + 140) == 12 )
  {
    do
      v7 = *(_QWORD *)(v7 + 160);
    while ( *(_BYTE *)(v7 + 140) == 12 );
  }
  result = sub_8D7820(v8, v7, a4 == 0, a4);
  if ( (_DWORD)result )
    return (unsigned int)sub_8DED30(v7, v8, 145, v10, v11) != 0;
  return result;
}
