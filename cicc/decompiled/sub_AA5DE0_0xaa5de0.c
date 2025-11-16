// Function: sub_AA5DE0
// Address: 0xaa5de0
//
void __fastcall sub_AA5DE0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // r15
  int v7; // r14d
  unsigned int v8; // ebx
  __int64 v9; // rsi
  __int64 v10; // rax

  v3 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v3 != a1 + 48 )
  {
    if ( !v3 )
      BUG();
    v4 = v3 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v3 - 24) - 30 <= 0xA )
    {
      v7 = sub_B46E30(v4);
      if ( v7 )
      {
        v8 = 0;
        do
        {
          v9 = v8++;
          v10 = sub_B46EC0(v4, v9);
          sub_AA5D60(v10, a2, a3);
        }
        while ( v7 != v8 );
      }
    }
  }
}
