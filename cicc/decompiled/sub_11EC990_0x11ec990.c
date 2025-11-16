// Function: sub_11EC990
// Address: 0x11ec990
//
char __fastcall sub_11EC990(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rbx
  __int64 v9; // r12
  __int64 v10; // rdi
  unsigned int v11; // r15d
  __int64 v13; // rdx
  __int64 v14; // rbx
  unsigned int v15; // esi
  char result; // al
  int v17; // eax
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // r12
  __int64 v20; // rax
  _QWORD *v21; // rcx
  unsigned __int64 v22; // rax
  _QWORD *v23; // rax
  unsigned int v24; // [rsp+0h] [rbp-50h]
  __int64 v25; // [rsp+10h] [rbp-40h] BYREF
  __int64 v26; // [rsp+18h] [rbp-38h]

  v8 = a3;
  v26 = a4;
  v25 = a5;
  v9 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( BYTE4(a6) )
  {
    v10 = *(_QWORD *)(a2 + 32 * ((unsigned int)a6 - v9));
    if ( *(_BYTE *)v10 != 17 )
      return 0;
    v11 = *(_DWORD *)(v10 + 32);
    if ( !(v11 <= 0x40 ? *(_QWORD *)(v10 + 24) == 0 : v11 == (unsigned int)sub_C444A0(v10 + 24)) )
      return 0;
  }
  v13 = (unsigned int)v26;
  v14 = *(_QWORD *)(a2 + 32 * (v8 - v9));
  if ( BYTE4(v26) && *(_QWORD *)(a2 + 32 * ((unsigned int)v26 - v9)) == v14 )
    return BYTE4(v26);
  if ( *(_BYTE *)v14 != 17 )
    return 0;
  v15 = *(_DWORD *)(v14 + 32);
  result = 1;
  if ( v15 )
  {
    if ( v15 <= 0x40 )
    {
      result = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v15) == *(_QWORD *)(v14 + 24);
    }
    else
    {
      v24 = *(_DWORD *)(v14 + 32);
      v17 = sub_C445E0(v14 + 24);
      v15 = v24;
      v13 = (unsigned int)v26;
      result = v24 == v17;
    }
    if ( !result )
    {
      if ( *(_BYTE *)(a1 + 8) )
        return 0;
      if ( BYTE4(v25) )
      {
        v18 = sub_98B430(*(_QWORD *)(a2 + 32 * ((unsigned int)v25 - v9)), 8u);
        v19 = v18;
        if ( !v18 )
          return 0;
        sub_11DA2E0(a2, (unsigned int *)&v25, 1, v18);
        v23 = *(_QWORD **)(v14 + 24);
        if ( *(_DWORD *)(v14 + 32) > 0x40u )
          v23 = (_QWORD *)*v23;
        return v19 <= (unsigned __int64)v23;
      }
      else
      {
        if ( !BYTE4(v26) )
          return 0;
        v20 = *(_QWORD *)(a2 + 32 * (v13 - v9));
        if ( *(_BYTE *)v20 != 17 )
          return 0;
        v21 = *(_QWORD **)(v14 + 24);
        if ( v15 > 0x40 )
          v21 = (_QWORD *)*v21;
        if ( *(_DWORD *)(v20 + 32) <= 0x40u )
          v22 = *(_QWORD *)(v20 + 24);
        else
          v22 = **(_QWORD **)(v20 + 24);
        return v22 <= (unsigned __int64)v21;
      }
    }
  }
  return result;
}
