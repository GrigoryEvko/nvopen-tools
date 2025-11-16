// Function: sub_1F49D30
// Address: 0x1f49d30
//
__int64 __fastcall sub_1F49D30(__int64 a1, int a2, __int64 a3)
{
  __int64 v5; // rax
  __int16 v6; // dx

  do
  {
    v5 = sub_1E69D00(a3, a2);
    v6 = **(_WORD **)(v5 + 16);
    if ( v6 == 15 )
    {
      a2 = *(_DWORD *)(*(_QWORD *)(v5 + 32) + 48LL);
    }
    else
    {
      if ( v6 != 10 )
        return (unsigned int)a2;
      a2 = *(_DWORD *)(*(_QWORD *)(v5 + 32) + 88LL);
    }
  }
  while ( a2 < 0 );
  return (unsigned int)a2;
}
