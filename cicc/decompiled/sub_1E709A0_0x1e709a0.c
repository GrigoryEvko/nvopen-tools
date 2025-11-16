// Function: sub_1E709A0
// Address: 0x1e709a0
//
__int64 __fastcall sub_1E709A0(__int64 a1)
{
  __int64 result; // rax
  __int64 *v3; // rdx
  __int64 v4; // rcx
  unsigned __int64 *v5; // rcx
  unsigned __int64 v6; // rdi
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 v9; // r10
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 *v13; // r8
  __int64 v14; // rax
  unsigned __int64 *v15; // rax
  unsigned __int64 v16; // rsi
  __int64 v17; // rax
  __int64 *v18; // rax
  __int64 v19; // rdx

  result = *(_QWORD *)(a1 + 2024);
  if ( result )
  {
    v3 = *(__int64 **)(a1 + 928);
    if ( (__int64 *)result != v3 )
    {
      v4 = *(_QWORD *)(a1 + 2024);
      if ( (*(_QWORD *)result & 4) == 0 && (*(_BYTE *)(result + 46) & 8) != 0 )
      {
        do
          v4 = *(_QWORD *)(v4 + 8);
        while ( (*(_BYTE *)(v4 + 46) & 8) != 0 );
      }
      v5 = *(unsigned __int64 **)(v4 + 8);
      if ( (unsigned __int64 *)result != v5 && v3 != (__int64 *)v5 && v5 != (unsigned __int64 *)result )
      {
        v6 = *v5;
        *(_QWORD *)((*(_QWORD *)result & 0xFFFFFFFFFFFFFFF8LL) + 8) = v5;
        v6 &= 0xFFFFFFFFFFFFFFF8LL;
        *v5 = *v5 & 7 | *(_QWORD *)result & 0xFFFFFFFFFFFFFFF8LL;
        v7 = *v3;
        *(_QWORD *)(v6 + 8) = v3;
        v7 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)result = v7 | *(_QWORD *)result & 7LL;
        *(_QWORD *)(v7 + 8) = result;
        *v3 = v6 | *v3 & 7;
        result = *(_QWORD *)(a1 + 2024);
      }
      v3 = (__int64 *)result;
    }
    *(_QWORD *)(a1 + 928) = v3;
  }
  v8 = *(_QWORD *)(a1 + 2008);
  v9 = *(_QWORD *)(a1 + 2000);
  if ( v8 != v9 )
  {
    do
    {
      v10 = *(_QWORD *)(a1 + 928);
      v11 = *(_QWORD *)(v8 - 16);
      v8 -= 16;
      v12 = *(_QWORD *)(v8 + 8);
      if ( v10 == v11 )
      {
        if ( !v10 )
          BUG();
        if ( (*(_BYTE *)v10 & 4) != 0 )
        {
          *(_QWORD *)(a1 + 928) = *(_QWORD *)(v10 + 8);
        }
        else
        {
          while ( (*(_BYTE *)(v10 + 46) & 8) != 0 )
            v10 = *(_QWORD *)(v10 + 8);
          *(_QWORD *)(a1 + 928) = *(_QWORD *)(v10 + 8);
        }
      }
      if ( !v12 )
        BUG();
      if ( (*(_BYTE *)v12 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v12 + 46) & 8) != 0 )
          v12 = *(_QWORD *)(v12 + 8);
      }
      v13 = *(__int64 **)(v12 + 8);
      if ( (__int64 *)v11 != v13 )
      {
        if ( !v11 )
          BUG();
        v14 = v11;
        if ( (*(_QWORD *)v11 & 4) == 0 && (*(_BYTE *)(v11 + 46) & 8) != 0 )
        {
          do
            v14 = *(_QWORD *)(v14 + 8);
          while ( (*(_BYTE *)(v14 + 46) & 8) != 0 );
        }
        v15 = *(unsigned __int64 **)(v14 + 8);
        if ( (unsigned __int64 *)v11 != v15 && v13 != (__int64 *)v15 && v15 != (unsigned __int64 *)v11 )
        {
          v16 = *v15;
          *(_QWORD *)((*(_QWORD *)v11 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v15;
          v16 &= 0xFFFFFFFFFFFFFFF8LL;
          *v15 = *v15 & 7 | *(_QWORD *)v11 & 0xFFFFFFFFFFFFFFF8LL;
          v17 = *v13;
          *(_QWORD *)(v16 + 8) = v13;
          v17 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)v11 = v17 | *(_QWORD *)v11 & 7LL;
          *(_QWORD *)(v17 + 8) = v11;
          *v13 = v16 | *v13 & 7;
        }
      }
      v18 = (__int64 *)(**(_QWORD **)(a1 + 936) & 0xFFFFFFFFFFFFFFF8LL);
      if ( !v18 )
        BUG();
      v19 = *v18;
      if ( (*v18 & 4) == 0 && (*(_BYTE *)((**(_QWORD **)(a1 + 936) & 0xFFFFFFFFFFFFFFF8LL) + 46) & 4) != 0 )
      {
        while ( 1 )
        {
          v18 = (__int64 *)(v19 & 0xFFFFFFFFFFFFFFF8LL);
          if ( (*(_BYTE *)((v19 & 0xFFFFFFFFFFFFFFF8LL) + 46) & 4) == 0 )
            break;
          v19 = *v18;
        }
      }
      if ( v13 == v18 )
        *(_QWORD *)(a1 + 936) = v11;
    }
    while ( v9 != v8 );
    result = *(_QWORD *)(a1 + 2000);
    if ( *(_QWORD *)(a1 + 2008) != result )
      *(_QWORD *)(a1 + 2008) = result;
  }
  *(_QWORD *)(a1 + 2024) = 0;
  return result;
}
