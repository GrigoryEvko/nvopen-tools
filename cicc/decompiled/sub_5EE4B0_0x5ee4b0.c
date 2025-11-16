// Function: sub_5EE4B0
// Address: 0x5ee4b0
//
void __fastcall sub_5EE4B0(__int64 a1)
{
  __int64 v1; // rcx
  __int64 v2; // rdx
  char v3; // al
  unsigned __int8 v4; // si
  __int64 i; // rax
  char v6; // cl
  __int64 j; // rdx
  char v8; // cl
  __int64 *v9; // rsi
  char v10; // al
  __int64 v11; // rsi
  __int64 *v12; // rdx
  __int64 v13; // rdx

  v1 = *(_QWORD *)(a1 + 88);
  v2 = *(_QWORD *)(v1 + 8);
  if ( *(_BYTE *)(a1 + 96) )
  {
    if ( !v2 )
      *(_BYTE *)(a1 + 96) = 0;
  }
  else
  {
    if ( !v2 )
      return;
    v3 = *(_BYTE *)(v1 + 80);
    if ( v3 == 16 )
    {
      v1 = **(_QWORD **)(v1 + 88);
      v3 = *(_BYTE *)(v1 + 80);
    }
    if ( v3 == 24 )
    {
      v1 = *(_QWORD *)(v1 + 88);
      v3 = *(_BYTE *)(v1 + 80);
    }
    if ( v3 == 20 )
    {
      i = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v1 + 88) + 176LL) + 152LL);
    }
    else
    {
      v4 = v3 - 10;
      i = 0;
      if ( v4 <= 1u )
      {
        for ( i = *(_QWORD *)(*(_QWORD *)(v1 + 88) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
          ;
      }
    }
    v6 = *(_BYTE *)(v2 + 80);
    if ( v6 == 16 )
    {
      v2 = **(_QWORD **)(v2 + 88);
      v6 = *(_BYTE *)(v2 + 80);
    }
    if ( v6 == 24 )
    {
      v2 = *(_QWORD *)(v2 + 88);
      v6 = *(_BYTE *)(v2 + 80);
    }
    if ( v6 == 20 )
    {
      j = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v2 + 88) + 176LL) + 152LL);
    }
    else
    {
      if ( (unsigned __int8)(v6 - 10) > 1u )
        return;
      for ( j = *(_QWORD *)(*(_QWORD *)(v2 + 88) + 152LL); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
        ;
    }
    v8 = j != 0 && i != 0;
    if ( v8 )
    {
      while ( *(_BYTE *)(i + 140) == 12 )
        i = *(_QWORD *)(i + 160);
      v9 = *(__int64 **)(i + 168);
      v10 = v8;
      if ( (*((_BYTE *)v9 + 21) & 1) == 0 )
      {
        v11 = *v9;
        v10 = 0;
        if ( v11 )
          v10 = *(_BYTE *)(v11 + 35) & 1;
      }
      while ( *(_BYTE *)(j + 140) == 12 )
        j = *(_QWORD *)(j + 160);
      v12 = *(__int64 **)(j + 168);
      if ( (*((_BYTE *)v12 + 21) & 1) == 0 )
      {
        v13 = *v12;
        v8 = 0;
        if ( v13 )
          v8 = *(_BYTE *)(v13 + 35) & 1;
      }
      if ( v8 != v10 )
        *(_BYTE *)(a1 + 96) = 1;
    }
  }
}
