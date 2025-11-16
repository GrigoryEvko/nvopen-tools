// Function: sub_16235A0
// Address: 0x16235a0
//
__int64 __fastcall sub_16235A0(__int64 *a1, int a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r8d
  __int64 v4; // rbx
  __int64 v5; // r12
  __int64 v7; // rsi
  unsigned __int8 *v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // rsi

  v2 = *((_DWORD *)a1 + 2);
  v3 = 0;
  if ( v2 )
  {
    v4 = *a1;
    v5 = *a1 + 16LL * v2 - 16;
    if ( *(_DWORD *)v5 == a2 )
    {
      v12 = v2 - 1;
      *((_DWORD *)a1 + 2) = v12;
      v13 = 16 * v12 + v4;
      v14 = *(_QWORD *)(v13 + 8);
      if ( v14 )
        sub_161E7C0(v13 + 8, v14);
    }
    else
    {
      if ( v4 == v5 )
        return v3;
      while ( *(_DWORD *)v4 != a2 )
      {
        v4 += 16;
        if ( v4 == v5 )
          return 0;
      }
      *(_DWORD *)v4 = *(_DWORD *)v5;
      if ( v5 + 8 != v4 + 8 )
      {
        v7 = *(_QWORD *)(v4 + 8);
        if ( v7 )
          sub_161E7C0(v4 + 8, v7);
        v8 = *(unsigned __int8 **)(v5 + 8);
        *(_QWORD *)(v4 + 8) = v8;
        if ( v8 )
        {
          sub_1623210(v5 + 8, v8, v4 + 8);
          *(_QWORD *)(v5 + 8) = 0;
        }
      }
      v9 = (unsigned int)(*((_DWORD *)a1 + 2) - 1);
      *((_DWORD *)a1 + 2) = v9;
      v10 = *a1 + 16 * v9;
      v11 = *(_QWORD *)(v10 + 8);
      if ( v11 )
        sub_161E7C0(v10 + 8, v11);
    }
    return 1;
  }
  return 0;
}
