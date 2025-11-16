// Function: sub_1A1E0D0
// Address: 0x1a1e0d0
//
bool __fastcall sub_1A1E0D0(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v6; // r13
  int v7; // eax
  __int64 v8; // rbx
  int i; // r12d
  __int64 v10; // rdx
  int v11; // eax
  __int64 v12; // rcx
  __int64 v13; // rdi
  char v14; // al
  int v15; // eax
  __int64 v16; // [rsp+0h] [rbp-40h]
  __int64 v17; // [rsp+8h] [rbp-38h]

  if ( *(_BYTE *)(a1 + 8) == 14 )
  {
    v3 = *(_QWORD *)(a1 + 24);
    if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 13 > 1 || !(unsigned __int8)sub_1A1E0D0(*(_QWORD *)(a1 + 24), a2) )
    {
      v4 = sub_127FA20(a2, v3);
      return 8 * sub_12BE0A0(a2, v3) != v4;
    }
  }
  else
  {
    v6 = sub_15A9930(a2, a1);
    v7 = *(_DWORD *)(a1 + 12);
    if ( !v7 )
      return 0;
    v8 = 0;
    v16 = (unsigned int)(v7 - 1);
    for ( i = 8 * *(_DWORD *)(v6 + 16); ; i = v10 )
    {
      v12 = 8 * v8;
      v13 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8 * v8);
      if ( (unsigned int)*(unsigned __int8 *)(v13 + 8) - 13 <= 1 )
      {
        v14 = sub_1A1E0D0(v13, a2);
        v12 = 8 * v8;
        if ( v14 )
          break;
      }
      if ( v8 == v16 )
      {
        v15 = *(_DWORD *)(a1 + 12);
        if ( !v15 )
          return 0;
        return i + (unsigned int)sub_127FA20(a2, *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL * (unsigned int)(v15 - 1))) < (unsigned __int64)(8LL * *(_QWORD *)v6);
      }
      v10 = 8LL * *(_QWORD *)(v6 + 8 * v8 + 24);
      if ( (_DWORD)v8 != -1 )
      {
        v17 = 8LL * *(_QWORD *)(v6 + 8 * v8 + 24);
        v11 = sub_127FA20(a2, *(_QWORD *)(*(_QWORD *)(a1 + 16) + v12));
        LODWORD(v10) = v17;
        if ( (unsigned int)v17 > v11 + i )
          return 1;
      }
      ++v8;
    }
  }
  return 1;
}
