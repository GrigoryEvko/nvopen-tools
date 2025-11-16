// Function: sub_32120E0
// Address: 0x32120e0
//
bool __fastcall sub_32120E0(char *a1)
{
  char v1; // al
  __int64 v2; // r12
  __int64 v3; // rbx
  unsigned __int8 v4; // al
  char v5; // al
  unsigned __int8 v6; // al
  __int64 v7; // r12
  unsigned __int64 v8; // rax
  unsigned __int8 v9; // al
  __int64 v10; // r12
  unsigned __int64 v12; // rax
  __int64 v13; // rdx

  v1 = *a1;
  if ( *a1 == 34 )
    return 1;
  v2 = (__int64)a1;
  v3 = 0x8000000010003LL;
  while ( 1 )
  {
    if ( v1 == 36 )
    {
      v4 = *(_BYTE *)(v2 - 16);
      if ( (v4 & 2) != 0 )
      {
        v2 = *(_QWORD *)(*(_QWORD *)(v2 - 32) + 24LL);
        if ( !v2 )
          return 0;
      }
      else
      {
        v2 = *(_QWORD *)(v2 - 16 - 8LL * ((v4 >> 2) & 0xF) + 24);
        if ( !v2 )
          return 0;
      }
    }
    v5 = *(_BYTE *)v2;
    if ( *(_BYTE *)v2 == 14 )
    {
      if ( (unsigned __int16)sub_AF18C0(v2) != 4 )
        return 1;
      v6 = *(_BYTE *)(v2 - 16);
      if ( (v6 & 2) != 0 )
        v7 = *(_QWORD *)(v2 - 32);
      else
        v7 = v2 - 16 - 8LL * ((v6 >> 2) & 0xF);
      v2 = *(_QWORD *)(v7 + 24);
      if ( !v2 )
        return 0;
      v5 = *(_BYTE *)v2;
    }
    if ( v5 != 13 )
      break;
    v8 = (unsigned int)sub_AF18C0(v2) - 15;
    if ( (unsigned __int16)v8 > 0x33u || !_bittest64(&v3, v8) )
    {
      v9 = *(_BYTE *)(v2 - 16);
      v10 = (v9 & 2) != 0 ? *(_QWORD *)(v2 - 32) : v2 - 16 - 8LL * ((v9 >> 2) & 0xF);
      v2 = *(_QWORD *)(v10 + 24);
      v1 = *(_BYTE *)v2;
      if ( *(_BYTE *)v2 != 34 )
        continue;
    }
    return 1;
  }
  v12 = *(unsigned int *)(v2 + 44);
  if ( (unsigned int)v12 <= 0x10 )
  {
    v13 = 82308;
    if ( _bittest64(&v13, v12) )
      return 1;
  }
  return (unsigned __int16)sub_AF18C0(v2) == 59;
}
