// Function: sub_2F7B040
// Address: 0x2f7b040
//
__int64 __fastcall sub_2F7B040(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  int v4; // esi

  v2 = *(_QWORD *)(a2 + 32);
  v3 = a2 + 24;
  if ( a2 + 24 == v2 )
  {
    sub_2F7AE10(a1, 0);
    return 0;
  }
  else
  {
    v4 = 0;
    do
    {
      v2 = *(_QWORD *)(v2 + 8);
      ++v4;
    }
    while ( v3 != v2 );
    sub_2F7AE10(a1, v4);
    return 0;
  }
}
