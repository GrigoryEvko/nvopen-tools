// Function: sub_72F5E0
// Address: 0x72f5e0
//
__int64 __fastcall sub_72F5E0(__int64 a1, __int64 a2, int a3, _DWORD *a4, _DWORD *a5, _DWORD *a6)
{
  __int64 i; // r12
  __int64 result; // rax
  int v12; // eax

  if ( a4 )
    *a4 = 0;
  if ( a5 )
    *a5 = 0;
  if ( a6 )
    *a6 = 0;
  while ( *(_BYTE *)(a2 + 140) == 12 )
    a2 = *(_QWORD *)(a2 + 160);
  for ( ; *(_BYTE *)(a1 + 140) == 12; a1 = *(_QWORD *)(a1 + 160) )
    ;
  for ( i = *(_QWORD *)(**(_QWORD **)(a1 + 168) + 8LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( a3 )
  {
    if ( !(unsigned int)sub_8D2FB0(i) )
      goto LABEL_15;
  }
  else if ( !(unsigned int)sub_8D3070(i) )
  {
    goto LABEL_15;
  }
  i = sub_8D46C0(i);
  if ( a4 )
    *a4 = 1;
LABEL_15:
  if ( !(unsigned int)sub_8D3A70(i) )
    return 0;
  if ( i != a2 && !(unsigned int)sub_8DED30(i, a2, 3) )
  {
    if ( unk_4D04344 && sub_8D5CE0(a2, i) )
    {
      if ( a6 )
        *a6 = 1;
      goto LABEL_18;
    }
    return 0;
  }
LABEL_18:
  result = 1;
  if ( a5 )
  {
    v12 = 0;
    if ( *(_BYTE *)(i + 140) == 12 )
      v12 = sub_8D4C10(i, 1);
    *a5 = v12;
    return 1;
  }
  return result;
}
