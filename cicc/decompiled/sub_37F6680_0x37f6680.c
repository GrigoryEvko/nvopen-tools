// Function: sub_37F6680
// Address: 0x37f6680
//
void __fastcall sub_37F6680(__int64 a1, unsigned __int64 a2, int *a3)
{
  _DWORD *v5; // rdi
  _DWORD *v6; // rax
  unsigned __int64 v7; // rdx
  int v8; // ecx
  _DWORD *v9; // rcx
  unsigned __int64 v10; // rsi
  int j; // edx
  _DWORD *v12; // rdx
  int v13; // ecx
  __int64 v14; // r13
  _DWORD *v15; // rax
  int v16; // ecx
  _DWORD *v17; // rdx
  _DWORD *i; // rsi

  v5 = *(_DWORD **)a1;
  if ( (__int64)(*(_QWORD *)(a1 + 16) - (_QWORD)v5) >> 2 < a2 )
  {
    if ( a2 > 0x1FFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    v14 = a2;
    if ( a2 )
    {
      v15 = (_DWORD *)sub_22077B0(4 * a2);
      v16 = *a3;
      v17 = &v15[v14];
      for ( i = v15; v15 != v17; ++v15 )
        *v15 = v16;
      v5 = *(_DWORD **)a1;
    }
    else
    {
      i = 0;
      v17 = 0;
    }
    *(_QWORD *)a1 = i;
    *(_QWORD *)(a1 + 8) = v17;
    *(_QWORD *)(a1 + 16) = v17;
    if ( v5 )
      j_j___libc_free_0((unsigned __int64)v5);
  }
  else
  {
    v6 = *(_DWORD **)(a1 + 8);
    v7 = v6 - v5;
    if ( a2 <= v7 )
    {
      v12 = v5;
      if ( a2 )
      {
        v12 = &v5[a2];
        v13 = *a3;
        if ( v5 != v12 )
        {
          do
            *v5++ = v13;
          while ( v12 != v5 );
          v6 = *(_DWORD **)(a1 + 8);
        }
      }
      if ( v6 != v12 )
        *(_QWORD *)(a1 + 8) = v12;
    }
    else
    {
      v8 = *a3;
      if ( v5 != v6 )
      {
        do
          *v5++ = v8;
        while ( v6 != v5 );
        v6 = *(_DWORD **)(a1 + 8);
        v7 = ((__int64)v6 - *(_QWORD *)a1) >> 2;
      }
      v9 = v6;
      v10 = a2 - v7;
      if ( v10 )
      {
        v9 = &v6[v10];
        for ( j = *a3; v9 != v6; ++v6 )
          *v6 = j;
      }
      *(_QWORD *)(a1 + 8) = v9;
    }
  }
}
