// Function: sub_3136DD0
// Address: 0x3136dd0
//
void __fastcall sub_3136DD0(__int64 a1, __int64 a2)
{
  int v2; // ecx
  _QWORD *v4; // rsi
  __int64 *v5; // rax
  __int64 v6; // rdx
  __int64 *v7; // rbx
  __int64 *v8; // r12
  __int64 v9; // rax
  __int64 *v10; // rdx
  __int64 *v11; // rax
  __int64 v12; // rdx

  v2 = *(_DWORD *)(a1 + 120);
  if ( v2 )
  {
    v4 = *(_QWORD **)(a1 + 112);
    if ( *v4 && *v4 != -8 )
    {
      v7 = *(__int64 **)(a1 + 112);
    }
    else
    {
      v5 = v4 + 1;
      do
      {
        do
        {
          v6 = *v5;
          v7 = v5++;
        }
        while ( v6 == -8 );
      }
      while ( !v6 );
    }
    v8 = &v4[v2];
    while ( v7 != v8 )
    {
      while ( 1 )
      {
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD, __int64))a2)(
          *(_QWORD *)(a2 + 8),
          *v7 + 96,
          *(_QWORD *)*v7,
          *v7 + 8);
        v9 = v7[1];
        v10 = v7 + 1;
        if ( !v9 || v9 == -8 )
          break;
        ++v7;
        if ( v10 == v8 )
          return;
      }
      v11 = v7 + 2;
      do
      {
        do
        {
          v12 = *v11;
          v7 = v11++;
        }
        while ( v12 == -8 );
      }
      while ( !v12 );
    }
  }
}
