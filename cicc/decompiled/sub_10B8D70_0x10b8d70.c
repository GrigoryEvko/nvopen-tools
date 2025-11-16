// Function: sub_10B8D70
// Address: 0x10b8d70
//
bool __fastcall sub_10B8D70(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v5; // r13
  const void **v6; // rsi
  unsigned int v7; // r12d
  bool result; // al
  unsigned int v9; // ebx
  char v10; // cl
  unsigned __int64 v11; // rbx
  unsigned int v12; // edx
  unsigned __int64 v13; // r12
  __int64 v14; // rdx
  _BYTE *v15; // rax
  __int64 v16; // rdx
  _BYTE *v17; // rax
  __int64 v18; // rdx
  _BYTE *v19; // rax

  v3 = a1 + 24;
  if ( *(_BYTE *)a1 != 17 )
  {
    v14 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a1 + 8) + 8LL) - 17;
    if ( (unsigned int)v14 > 1 )
      return 0;
    if ( *(_BYTE *)a1 > 0x15u )
      return 0;
    v15 = sub_AD7630(a1, 1, v14);
    if ( !v15 || *v15 != 17 )
      return 0;
    v3 = (__int64)(v15 + 24);
  }
  if ( *(_BYTE *)a2 == 17 )
  {
    v5 = a2 + 24;
  }
  else
  {
    v16 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a2 + 8) + 8LL) - 17;
    if ( (unsigned int)v16 > 1 )
      return 0;
    if ( *(_BYTE *)a2 > 0x15u )
      return 0;
    v17 = sub_AD7630(a2, 0, v16);
    if ( !v17 || *v17 != 17 )
      return 0;
    v5 = (__int64)(v17 + 24);
  }
  v6 = (const void **)(a3 + 24);
  if ( *(_BYTE *)a3 != 17 )
  {
    v18 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a3 + 8) + 8LL) - 17;
    if ( (unsigned int)v18 > 1 )
      return 0;
    if ( *(_BYTE *)a3 > 0x15u )
      return 0;
    v19 = sub_AD7630(a3, 0, v18);
    if ( !v19 || *v19 != 17 )
      return 0;
    v6 = (const void **)(v19 + 24);
  }
  v7 = *(_DWORD *)(v5 + 8);
  if ( v7 > 0x40 )
  {
    if ( sub_C43C50(v5, v6) )
      goto LABEL_7;
    return 0;
  }
  if ( *(const void **)v5 != *v6 )
    return 0;
LABEL_7:
  result = 1;
  if ( *(_BYTE *)a1 != 13 )
  {
    v9 = *(_DWORD *)(v3 + 8);
    if ( v9 > 0x40 )
    {
      v9 = sub_C44500(v3);
    }
    else if ( v9 )
    {
      v10 = 64 - v9;
      v9 = 64;
      if ( *(_QWORD *)v3 << v10 != -1 )
      {
        _BitScanReverse64(&v11, ~(*(_QWORD *)v3 << v10));
        v9 = v11 ^ 0x3F;
      }
    }
    if ( v7 > 0x40 )
    {
      v7 = sub_C444A0(v5);
    }
    else
    {
      v12 = v7 - 64;
      if ( *(_QWORD *)v5 )
      {
        _BitScanReverse64(&v13, *(_QWORD *)v5);
        v7 = v12 + (v13 ^ 0x3F);
      }
    }
    return v7 == v9;
  }
  return result;
}
