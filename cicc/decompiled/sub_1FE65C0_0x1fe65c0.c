// Function: sub_1FE65C0
// Address: 0x1fe65c0
//
__int64 __fastcall sub_1FE65C0(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // rax
  int v4; // r8d
  __int64 v6; // rax
  __int64 v7; // rax

  v3 = *(_QWORD *)(a2 + 48);
  v4 = 0;
  if ( v3 )
  {
    if ( !*(_QWORD *)(v3 + 32) )
    {
      v6 = *(_QWORD *)(v3 + 16);
      if ( *(_WORD *)(v6 + 24) == 46 )
      {
        v7 = *(_QWORD *)(v6 + 32);
        if ( a2 == *(_QWORD *)(v7 + 80) && a3 == *(_DWORD *)(v7 + 88) )
        {
          v4 = *(_DWORD *)(*(_QWORD *)(v7 + 40) + 84LL);
          if ( v4 >= 0 )
            return 0;
        }
      }
    }
  }
  return (unsigned int)v4;
}
