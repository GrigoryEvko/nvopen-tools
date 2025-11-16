// Function: sub_210DAB0
// Address: 0x210dab0
//
__int64 __fastcall sub_210DAB0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rdi
  int v5; // esi

  v2 = *(_QWORD *)(a2 + 32);
  v3 = a2 + 24;
  v4 = a1 + 160;
  if ( a2 + 24 == v2 )
  {
    sub_210D890(v4, 0);
    return 0;
  }
  else
  {
    v5 = 0;
    do
    {
      v2 = *(_QWORD *)(v2 + 8);
      ++v5;
    }
    while ( v3 != v2 );
    sub_210D890(v4, v5);
    return 0;
  }
}
