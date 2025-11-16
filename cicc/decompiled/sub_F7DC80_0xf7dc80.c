// Function: sub_F7DC80
// Address: 0xf7dc80
//
__int64 __fastcall sub_F7DC80(__int64 a1, unsigned __int8 *a2, unsigned __int8 *a3, __int64 a4)
{
  int v6; // eax
  __int64 v7; // rdx
  unsigned __int8 *v8; // rax
  __int64 v9; // rax
  unsigned __int64 v10; // rdx

  do
  {
    v9 = sub_D4B130(a4);
    v10 = *(_QWORD *)(v9 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v10 == v9 + 48 )
    {
      v7 = 0;
    }
    else
    {
      if ( !v10 )
        BUG();
      v6 = *(unsigned __int8 *)(v10 - 24);
      v7 = v10 - 24;
      if ( (unsigned int)(v6 - 30) >= 0xB )
        v7 = 0;
    }
    v8 = sub_F7D9C0(a1, a3, v7, 0);
    a3 = v8;
    if ( !v8 )
      return 0;
  }
  while ( v8 != a2 );
  return 1;
}
