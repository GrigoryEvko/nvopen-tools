// Function: sub_2A37EC0
// Address: 0x2a37ec0
//
bool __fastcall sub_2A37EC0(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int8 v2; // dl
  __int64 v3; // r12
  __int64 *v4; // rbx
  __int64 v5; // r12
  __int64 *v6; // r14
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 *v9; // r12
  _BYTE *v10; // rdi
  _BYTE *v11; // rdi
  _BYTE *v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rdx
  bool result; // al
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx

  if ( (*(_BYTE *)(a1 + 7) & 0x20) == 0 )
    return 0;
  if ( !sub_B91C10(a1, 30) )
    return 0;
  if ( (*(_BYTE *)(a1 + 7) & 0x20) == 0 )
    BUG();
  v1 = sub_B91C10(a1, 30);
  v2 = *(_BYTE *)(v1 - 16);
  if ( (v2 & 2) != 0 )
  {
    v4 = *(__int64 **)(v1 - 32);
    v3 = *(unsigned int *)(v1 - 24);
  }
  else
  {
    v3 = (*(_WORD *)(v1 - 16) >> 6) & 0xF;
    v4 = (__int64 *)(v1 - 8LL * ((v2 >> 2) & 0xF) - 16);
  }
  v5 = 8 * v3;
  v6 = &v4[(unsigned __int64)v5 / 8];
  v7 = v5 >> 3;
  v8 = v5 >> 5;
  if ( v8 )
  {
    v9 = &v4[4 * v8];
    while ( 1 )
    {
      if ( !*(_BYTE *)*v4 )
      {
        v13 = sub_B91420(*v4);
        if ( v14 == 9 && *(_QWORD *)v13 == 0x696E692D6F747561LL && *(_BYTE *)(v13 + 8) == 116 )
          return v6 != v4;
      }
      v10 = (_BYTE *)v4[1];
      if ( !*v10 )
      {
        v16 = sub_B91420((__int64)v10);
        if ( v17 == 9 && *(_QWORD *)v16 == 0x696E692D6F747561LL && *(_BYTE *)(v16 + 8) == 116 )
          return v6 != v4 + 1;
      }
      v11 = (_BYTE *)v4[2];
      if ( !*v11 )
      {
        v18 = sub_B91420((__int64)v11);
        if ( v19 == 9 && *(_QWORD *)v18 == 0x696E692D6F747561LL && *(_BYTE *)(v18 + 8) == 116 )
          return v6 != v4 + 2;
      }
      v12 = (_BYTE *)v4[3];
      if ( !*v12 )
      {
        v20 = sub_B91420((__int64)v12);
        if ( v21 == 9 && *(_QWORD *)v20 == 0x696E692D6F747561LL && *(_BYTE *)(v20 + 8) == 116 )
          return v6 != v4 + 3;
      }
      v4 += 4;
      if ( v9 == v4 )
      {
        v7 = v6 - v4;
        break;
      }
    }
  }
  if ( v7 != 2 )
  {
    if ( v7 != 3 )
    {
      if ( v7 != 1 )
        return 0;
      goto LABEL_40;
    }
    if ( !*(_BYTE *)*v4 )
    {
      v26 = sub_B91420(*v4);
      if ( v27 == 9 && *(_QWORD *)v26 == 0x696E692D6F747561LL && *(_BYTE *)(v26 + 8) == 116 )
        return v4 != v6;
    }
    ++v4;
  }
  if ( !*(_BYTE *)*v4 )
  {
    v24 = sub_B91420(*v4);
    if ( v25 == 9 && *(_QWORD *)v24 == 0x696E692D6F747561LL && *(_BYTE *)(v24 + 8) == 116 )
      return v6 != v4;
  }
  ++v4;
LABEL_40:
  result = 0;
  if ( !*(_BYTE *)*v4 )
  {
    v22 = sub_B91420(*v4);
    if ( v23 != 9 || *(_QWORD *)v22 != 0x696E692D6F747561LL || *(_BYTE *)(v22 + 8) != 116 )
      return 0;
    return v6 != v4;
  }
  return result;
}
