// Function: sub_349EDA0
// Address: 0x349eda0
//
__int64 __fastcall sub_349EDA0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r8
  __int64 v6; // rdx
  __int64 v7; // rax
  int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v13; // r8
  __int64 v14; // r8
  __int64 v15; // r9

  v5 = a1;
  v6 = (a2 - a1) >> 7;
  v7 = (a2 - a1) >> 5;
  if ( v6 > 0 )
  {
    v8 = *(_DWORD *)a3;
    v9 = a1 + (v6 << 7);
    while ( 1 )
    {
      if ( *(_DWORD *)v5 != v8 )
        goto LABEL_7;
      if ( v8 == 2 )
      {
        if ( *(_DWORD *)(v5 + 8) == *(_DWORD *)(a3 + 8)
          && *(_QWORD *)(v5 + 16) == *(_QWORD *)(a3 + 16)
          && *(_QWORD *)(v5 + 24) == *(_QWORD *)(a3 + 24) )
        {
          return v5;
        }
      }
      else
      {
        if ( v8 <= 2 )
        {
          if ( v8 != 1 )
            goto LABEL_78;
LABEL_33:
          if ( *(_QWORD *)(v5 + 8) == *(_QWORD *)(a3 + 8) )
            return v5;
          goto LABEL_7;
        }
        if ( v8 == 3 )
          goto LABEL_33;
        if ( v8 != 4 )
          goto LABEL_78;
        if ( *(_DWORD *)(v5 + 8) == *(_DWORD *)(a3 + 8) && *(_QWORD *)(v5 + 16) == *(_QWORD *)(a3 + 16) )
          return v5;
      }
LABEL_7:
      v10 = v5 + 32;
      if ( v8 != *(_DWORD *)(v5 + 32) )
        goto LABEL_13;
      if ( v8 == 2 )
      {
        if ( *(_DWORD *)(v5 + 40) == *(_DWORD *)(a3 + 8)
          && *(_QWORD *)(v5 + 48) == *(_QWORD *)(a3 + 16)
          && *(_QWORD *)(v5 + 56) == *(_QWORD *)(a3 + 24) )
        {
          return v10;
        }
      }
      else
      {
        if ( v8 <= 2 )
        {
          if ( v8 != 1 )
            goto LABEL_78;
LABEL_36:
          if ( *(_QWORD *)(v5 + 40) == *(_QWORD *)(a3 + 8) )
            return v10;
          goto LABEL_13;
        }
        if ( v8 == 3 )
          goto LABEL_36;
        if ( v8 != 4 )
          goto LABEL_78;
        if ( *(_DWORD *)(v5 + 40) == *(_DWORD *)(a3 + 8) && *(_QWORD *)(v5 + 48) == *(_QWORD *)(a3 + 16) )
          return v5 + 32;
      }
LABEL_13:
      v10 = v5 + 64;
      if ( v8 != *(_DWORD *)(v5 + 64) )
        goto LABEL_19;
      if ( v8 == 2 )
      {
        if ( *(_DWORD *)(v5 + 72) == *(_DWORD *)(a3 + 8)
          && *(_QWORD *)(v5 + 80) == *(_QWORD *)(a3 + 16)
          && *(_QWORD *)(v5 + 88) == *(_QWORD *)(a3 + 24) )
        {
          return v10;
        }
      }
      else
      {
        if ( v8 <= 2 )
        {
          if ( v8 != 1 )
            goto LABEL_78;
LABEL_40:
          if ( *(_QWORD *)(v5 + 72) == *(_QWORD *)(a3 + 8) )
            return v10;
          goto LABEL_19;
        }
        if ( v8 == 3 )
          goto LABEL_40;
        if ( v8 != 4 )
          goto LABEL_78;
        if ( *(_DWORD *)(v5 + 72) == *(_DWORD *)(a3 + 8) && *(_QWORD *)(v5 + 80) == *(_QWORD *)(a3 + 16) )
          return v5 + 64;
      }
LABEL_19:
      v10 = v5 + 96;
      if ( v8 != *(_DWORD *)(v5 + 96) )
        goto LABEL_25;
      if ( v8 == 2 )
      {
        if ( *(_DWORD *)(v5 + 104) == *(_DWORD *)(a3 + 8)
          && *(_QWORD *)(v5 + 112) == *(_QWORD *)(a3 + 16)
          && *(_QWORD *)(v5 + 120) == *(_QWORD *)(a3 + 24) )
        {
          return v10;
        }
        goto LABEL_25;
      }
      if ( v8 <= 2 )
      {
        if ( v8 != 1 )
LABEL_78:
          BUG();
        goto LABEL_43;
      }
      if ( v8 == 3 )
      {
LABEL_43:
        if ( *(_QWORD *)(v5 + 104) == *(_QWORD *)(a3 + 8) )
          return v10;
        v5 += 128;
        if ( v5 == v9 )
        {
LABEL_45:
          v7 = (a2 - v5) >> 5;
          break;
        }
      }
      else
      {
        if ( v8 != 4 )
          goto LABEL_78;
        if ( *(_DWORD *)(v5 + 104) == *(_DWORD *)(a3 + 8) && *(_QWORD *)(v5 + 112) == *(_QWORD *)(a3 + 16) )
          return v5 + 96;
LABEL_25:
        v5 += 128;
        if ( v5 == v9 )
          goto LABEL_45;
      }
    }
  }
  if ( v7 == 2 )
  {
LABEL_73:
    if ( sub_349D790(v5, a3) )
      return v13;
    v5 = v13 + 32;
    goto LABEL_75;
  }
  if ( v7 == 3 )
  {
    if ( sub_349D790(v5, a3) )
      return v13;
    v5 = v13 + 32;
    goto LABEL_73;
  }
  if ( v7 != 1 )
    return a2;
LABEL_75:
  if ( !sub_349D790(v5, a3) )
    return v15;
  return v14;
}
