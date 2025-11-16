// Function: sub_32642F0
// Address: 0x32642f0
//
bool __fastcall sub_32642F0(int a1, __int64 a2)
{
  __int64 v4; // rdi
  bool result; // al
  unsigned int v6; // ebx
  __int64 v7; // rdi
  unsigned int v8; // ebx
  __int64 v9; // rdi
  unsigned int v10; // eax
  __int64 v11; // rsi
  unsigned int v12; // r13d
  int v13; // r8d
  __int64 v14; // rdi
  unsigned int v15; // r13d
  unsigned int v16; // eax
  __int64 v17; // rsi
  __int64 v18; // rdi
  unsigned int v19; // edx
  __int64 v20; // r8
  unsigned int v21; // ebx
  unsigned int v22; // eax

  switch ( a1 )
  {
    case 12:
      goto LABEL_9;
    case 20:
      v9 = *(_QWORD *)(a2 + 96);
      v10 = *(_DWORD *)(v9 + 32);
      v11 = *(_QWORD *)(v9 + 24);
      v12 = v10 - 1;
      if ( v10 <= 0x40 )
      {
        result = 1;
        if ( 1LL << v12 == v11 )
          return result;
      }
      else if ( (*(_QWORD *)(v11 + 8LL * (v12 >> 6)) & (1LL << v12)) != 0 )
      {
        v13 = sub_C44590(v9 + 24);
        result = 1;
        if ( v13 == v12 )
          return result;
      }
      goto LABEL_26;
    case 10:
      v14 = *(_QWORD *)(a2 + 96);
      result = 1;
      v15 = *(_DWORD *)(v14 + 32);
      if ( !v15 )
        return result;
      result = v15 <= 0x40
             ? 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v15) == *(_QWORD *)(v14 + 24)
             : v15 == (unsigned int)sub_C445E0(v14 + 24);
      if ( result )
        return result;
      goto LABEL_26;
    case 18:
      v4 = *(_QWORD *)(a2 + 96);
      v22 = *(_DWORD *)(v4 + 32);
      v17 = *(_QWORD *)(v4 + 24);
      v6 = v22 - 1;
      if ( v22 > 0x40 )
      {
        result = 0;
        if ( (*(_QWORD *)(v17 + 8LL * (v6 >> 6)) & (1LL << v6)) != 0 )
          return result;
        return (unsigned int)sub_C445E0(v4 + 24) == v6;
      }
      return (1LL << v6) - 1 == v17;
  }
  if ( a1 != 13 )
  {
    if ( a1 == 21 )
    {
      v4 = *(_QWORD *)(a2 + 96);
      v16 = *(_DWORD *)(v4 + 32);
      v17 = *(_QWORD *)(v4 + 24);
      v6 = v16 - 1;
      if ( v16 > 0x40 )
      {
        result = 0;
        if ( (*(_QWORD *)(v17 + 8LL * (v6 >> 6)) & (1LL << v6)) != 0 )
          return result;
        return (unsigned int)sub_C445E0(v4 + 24) == v6;
      }
      return (1LL << v6) - 1 == v17;
    }
    if ( a1 != 11 )
    {
LABEL_26:
      result = 0;
      if ( a1 == 19 )
      {
        v18 = *(_QWORD *)(a2 + 96);
        v19 = *(_DWORD *)(v18 + 32);
        v20 = *(_QWORD *)(v18 + 24);
        v21 = v19 - 1;
        if ( v19 <= 0x40 )
        {
          return v20 == 1LL << v21;
        }
        else if ( (*(_QWORD *)(v20 + 8LL * (v21 >> 6)) & (1LL << v21)) != 0 )
        {
          return v21 == (unsigned int)sub_C44590(v18 + 24);
        }
      }
      return result;
    }
LABEL_9:
    v7 = *(_QWORD *)(a2 + 96);
    v8 = *(_DWORD *)(v7 + 32);
    if ( v8 <= 0x40 )
      return *(_QWORD *)(v7 + 24) == 0;
    else
      return v8 == (unsigned int)sub_C444A0(v7 + 24);
  }
  v4 = *(_QWORD *)(a2 + 96);
  result = 1;
  v6 = *(_DWORD *)(v4 + 32);
  if ( v6 )
  {
    if ( v6 <= 0x40 )
      return *(_QWORD *)(v4 + 24) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v6);
    return (unsigned int)sub_C445E0(v4 + 24) == v6;
  }
  return result;
}
