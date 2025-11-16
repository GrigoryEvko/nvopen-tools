// Function: sub_A45A10
// Address: 0xa45a10
//
void __fastcall sub_A45A10(__int64 a1, __int64 a2)
{
  int v2; // ecx
  __int64 *v3; // rsi
  __int64 *v4; // rax
  __int64 v5; // rdx
  __int64 *v6; // rbx
  __int64 *v7; // r13
  __int64 v8; // rax
  __int64 *v9; // rdx
  __int64 *v10; // rax
  __int64 v11; // rdx

  v2 = *(_DWORD *)(a2 + 8);
  if ( v2 )
  {
    v3 = *(__int64 **)a2;
    if ( *v3 && *v3 != -8 )
    {
      v6 = v3;
    }
    else
    {
      v4 = v3 + 1;
      do
      {
        do
        {
          v5 = *v4;
          v6 = v4++;
        }
        while ( v5 == -8 );
      }
      while ( !v5 );
    }
    v7 = &v3[v2];
    while ( v6 != v7 )
    {
      while ( 1 )
      {
        sub_A45280(a1, *(unsigned __int8 **)(*v6 + 8));
        v8 = v6[1];
        v9 = v6 + 1;
        if ( !v8 || v8 == -8 )
          break;
        ++v6;
        if ( v9 == v7 )
          return;
      }
      v10 = v6 + 2;
      do
      {
        do
        {
          v11 = *v10;
          v6 = v10++;
        }
        while ( v11 == -8 );
      }
      while ( !v11 );
    }
  }
}
