// Function: sub_BDBB60
// Address: 0xbdbb60
//
__int64 __fastcall sub_BDBB60(__int64 a1, int a2)
{
  __int64 v2; // r14
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r14
  __int64 v6; // r13
  __int64 v7; // rbx
  __int64 v8; // r13
  __int64 v9; // rax
  bool v10; // al

  LODWORD(v2) = 0;
  if ( *(char *)(a1 + 7) < 0 )
  {
    v3 = sub_BD2BC0(a1);
    v5 = v3 + v4;
    if ( *(char *)(a1 + 7) < 0 )
      v5 -= sub_BD2BC0(a1);
    v2 = v5 >> 4;
    if ( (_DWORD)v2 )
    {
      v6 = (unsigned int)v2;
      v7 = 0;
      LODWORD(v2) = 0;
      v8 = 16 * v6;
      do
      {
        v9 = 0;
        if ( *(char *)(a1 + 7) < 0 )
          v9 = sub_BD2BC0(a1);
        v10 = a2 == *(_DWORD *)(*(_QWORD *)(v9 + v7) + 8LL);
        v7 += 16;
        LODWORD(v2) = v10 + (_DWORD)v2;
      }
      while ( v8 != v7 );
    }
  }
  return (unsigned int)v2;
}
