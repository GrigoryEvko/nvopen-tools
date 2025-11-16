// Function: sub_1EA9F10
// Address: 0x1ea9f10
//
unsigned __int64 *__fastcall sub_1EA9F10(__int64 a1)
{
  __int64 *v2; // rcx
  __int64 v3; // rax
  __int64 v4; // rdx
  unsigned __int64 *v5; // rdx
  unsigned __int64 v6; // rdi
  __int64 v7; // rsi
  unsigned __int64 *result; // rax
  __int64 v9; // r13
  __int64 v10; // rbx
  unsigned __int64 v11; // rdx
  unsigned __int64 *v12; // rcx
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rdx
  __int64 *v17; // rdx
  __int64 *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 *v21; // rsi
  __int64 v22; // rax
  unsigned __int64 *v23; // rax
  unsigned __int64 v24; // r9
  __int64 v25; // rax

  v2 = *(__int64 **)(a1 + 936);
  v3 = *(_QWORD *)(a1 + 2024);
  *(_QWORD *)(a1 + 928) = v2;
  if ( v3 && (__int64 *)v3 != v2 )
  {
    v4 = v3;
    if ( (*(_QWORD *)v3 & 4) == 0 && (*(_BYTE *)(v3 + 46) & 8) != 0 )
    {
      do
        v4 = *(_QWORD *)(v4 + 8);
      while ( (*(_BYTE *)(v4 + 46) & 8) != 0 );
    }
    v5 = *(unsigned __int64 **)(v4 + 8);
    if ( (unsigned __int64 *)v3 != v5 && v2 != (__int64 *)v5 && v5 != (unsigned __int64 *)v3 )
    {
      v6 = *v5;
      *(_QWORD *)((*(_QWORD *)v3 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v5;
      v6 &= 0xFFFFFFFFFFFFFFF8LL;
      *v5 = *v5 & 7 | *(_QWORD *)v3 & 0xFFFFFFFFFFFFFFF8LL;
      v7 = *v2;
      *(_QWORD *)(v6 + 8) = v2;
      v7 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v3 = v7 | *(_QWORD *)v3 & 7LL;
      *(_QWORD *)(v7 + 8) = v3;
      *v2 = v6 | *v2 & 7;
    }
  }
  result = *(unsigned __int64 **)(a1 + 2232);
  v9 = (__int64)(*(_QWORD *)(a1 + 2240) - (_QWORD)result) >> 3;
  if ( (_DWORD)v9 )
  {
    v10 = 0;
    result = (unsigned __int64 *)*result;
    if ( result )
      goto LABEL_12;
LABEL_24:
    result = (unsigned __int64 *)(*(__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 16) + 632LL))(
                                   *(_QWORD *)(a1 + 16),
                                   *(_QWORD *)(a1 + 920),
                                   *(_QWORD *)(a1 + 936));
    if ( v10 )
    {
      while ( ++v10 != (unsigned int)v9 )
      {
LABEL_23:
        result = *(unsigned __int64 **)(*(_QWORD *)(a1 + 2232) + 8 * v10);
        if ( !result )
          goto LABEL_24;
LABEL_12:
        v11 = result[1];
        v12 = *(unsigned __int64 **)(a1 + 936);
        if ( (unsigned __int64 *)v11 != v12 )
        {
          if ( !v11 )
            BUG();
          v13 = result[1];
          if ( (*(_QWORD *)v11 & 4) == 0 && (*(_BYTE *)(v11 + 46) & 8) != 0 )
          {
            do
              v13 = *(_QWORD *)(v13 + 8);
            while ( (*(_BYTE *)(v13 + 46) & 8) != 0 );
          }
          result = *(unsigned __int64 **)(v13 + 8);
          if ( (unsigned __int64 *)v11 != result && v12 != result && result != (unsigned __int64 *)v11 )
          {
            v14 = *result;
            *(_QWORD *)((*(_QWORD *)v11 & 0xFFFFFFFFFFFFFFF8LL) + 8) = result;
            v14 &= 0xFFFFFFFFFFFFFFF8LL;
            *result = *result & 7 | *(_QWORD *)v11 & 0xFFFFFFFFFFFFFFF8LL;
            v15 = *v12;
            *(_QWORD *)(v14 + 8) = v12;
            v15 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)v11 = v15 | *(_QWORD *)v11 & 7LL;
            *(_QWORD *)(v15 + 8) = v11;
            result = (unsigned __int64 *)(v14 | *v12 & 7);
            *v12 = (unsigned __int64)result;
          }
        }
        if ( !v10 )
          goto LABEL_25;
      }
    }
    else
    {
LABEL_25:
      result = (unsigned __int64 *)(**(_QWORD **)(a1 + 936) & 0xFFFFFFFFFFFFFFF8LL);
      if ( !result )
        BUG();
      v16 = *result;
      if ( (*result & 4) == 0 && (*(_BYTE *)((**(_QWORD **)(a1 + 936) & 0xFFFFFFFFFFFFFFF8LL) + 46) & 4) != 0 )
      {
        while ( 1 )
        {
          result = (unsigned __int64 *)(v16 & 0xFFFFFFFFFFFFFFF8LL);
          if ( (*(_BYTE *)((v16 & 0xFFFFFFFFFFFFFFF8LL) + 46) & 4) == 0 )
            break;
          v16 = *result;
        }
      }
      ++v10;
      *(_QWORD *)(a1 + 928) = result;
      if ( v10 != (unsigned int)v9 )
        goto LABEL_23;
    }
  }
  v17 = *(__int64 **)(a1 + 2008);
  v18 = *(__int64 **)(a1 + 2000);
  if ( v18 != v17 )
  {
    do
    {
      v19 = *(v17 - 1);
      v17 -= 2;
      v20 = *v17;
      if ( !v19 )
        BUG();
      if ( (*(_BYTE *)v19 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v19 + 46) & 8) != 0 )
          v19 = *(_QWORD *)(v19 + 8);
      }
      v21 = *(__int64 **)(v19 + 8);
      if ( (__int64 *)v20 != v21 )
      {
        if ( !v20 )
          BUG();
        v22 = *v17;
        if ( (*(_QWORD *)v20 & 4) == 0 && (*(_BYTE *)(v20 + 46) & 8) != 0 )
        {
          do
            v22 = *(_QWORD *)(v22 + 8);
          while ( (*(_BYTE *)(v22 + 46) & 8) != 0 );
        }
        v23 = *(unsigned __int64 **)(v22 + 8);
        if ( (unsigned __int64 *)v20 != v23 && v21 != (__int64 *)v23 && v23 != (unsigned __int64 *)v20 )
        {
          v24 = *v23;
          *(_QWORD *)((*(_QWORD *)v20 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v23;
          v24 &= 0xFFFFFFFFFFFFFFF8LL;
          *v23 = *v23 & 7 | *(_QWORD *)v20 & 0xFFFFFFFFFFFFFFF8LL;
          v25 = *v21;
          *(_QWORD *)(v24 + 8) = v21;
          v25 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)v20 = v25 | *(_QWORD *)v20 & 7LL;
          *(_QWORD *)(v25 + 8) = v20;
          *v21 = v24 | *v21 & 7;
        }
      }
    }
    while ( v18 != v17 );
    result = *(unsigned __int64 **)(a1 + 2000);
    if ( result != *(unsigned __int64 **)(a1 + 2008) )
      *(_QWORD *)(a1 + 2008) = result;
  }
  *(_QWORD *)(a1 + 2024) = 0;
  return result;
}
