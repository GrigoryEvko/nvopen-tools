// Function: sub_2FD76C0
// Address: 0x2fd76c0
//
__int64 __fastcall sub_2FD76C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  unsigned __int64 v5; // rbx
  __int64 v6; // r15
  unsigned int v7; // eax

  v4 = a2 + 48;
  v5 = *(_QWORD *)(a2 + 56);
  v6 = *(_QWORD *)(a2 + 32);
  if ( v5 != a2 + 48 )
  {
    do
    {
      while ( 1 )
      {
        if ( sub_2E88F60(v5) )
          sub_2E79700(v6, v5);
        if ( !v5 )
          BUG();
        if ( (*(_BYTE *)v5 & 4) == 0 )
          break;
        v5 = *(_QWORD *)(v5 + 8);
        if ( v4 == v5 )
          goto LABEL_9;
      }
      while ( (*(_BYTE *)(v5 + 44) & 8) != 0 )
        v5 = *(_QWORD *)(v5 + 8);
      v5 = *(_QWORD *)(v5 + 8);
    }
    while ( v4 != v5 );
  }
LABEL_9:
  if ( a3 )
    (*(void (__fastcall **)(_QWORD, __int64))a3)(*(_QWORD *)(a3 + 8), a2);
  while ( 1 )
  {
    v7 = *(_DWORD *)(a2 + 120);
    if ( !v7 )
      break;
    sub_2E33590(a2, (__int64 *)(*(_QWORD *)(a2 + 112) + 8LL * v7 - 8), 0);
  }
  return sub_2E32710((_QWORD *)a2);
}
