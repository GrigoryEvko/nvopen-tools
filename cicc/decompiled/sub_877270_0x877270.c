// Function: sub_877270
// Address: 0x877270
//
unsigned __int64 __fastcall sub_877270(_BYTE *a1, _QWORD *a2, char a3, unsigned __int64 a4)
{
  _QWORD *v4; // r15
  unsigned __int64 v6; // r14
  const char *v7; // rdx
  int v8; // eax

  v4 = a2;
  if ( a4 > 1 && a2 != 0 && (a3 & 1) != 0 )
  {
    *a1 = 58;
    v6 = 1;
  }
  else
  {
    v6 = 0;
    if ( !a2 )
      return v6;
  }
  if ( a4 - 1 > v6 )
  {
    do
    {
      v7 = (const char *)&unk_3C1ECA8;
      if ( !v4[1] )
        v7 = "%s";
      v8 = snprintf(&a1[v6], a4 - v6, v7, *(_QWORD *)(*v4 + 8LL));
      v4 = (_QWORD *)v4[1];
      v6 += v8;
    }
    while ( v4 && v6 < a4 - 1 );
  }
  return v6;
}
