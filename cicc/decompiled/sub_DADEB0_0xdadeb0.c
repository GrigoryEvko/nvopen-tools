// Function: sub_DADEB0
// Address: 0xdadeb0
//
__int64 __fastcall sub_DADEB0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v4; // r14
  __int64 v5; // rdx
  char v6; // r12
  __int64 result; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  _QWORD *v10; // rdx
  _QWORD *v11; // rax
  __int64 *v12; // rax
  __int64 *v13; // r13
  __int64 *v14; // r12
  __int64 v15; // rsi
  _QWORD *v16; // rax
  _QWORD *v17; // rdx
  __int64 *v18; // [rsp-40h] [rbp-40h]

  switch ( *(_WORD *)(a2 + 24) )
  {
    case 0:
    case 1:
      return 1;
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
    case 9:
    case 0xA:
    case 0xB:
    case 0xC:
    case 0xD:
    case 0xE:
      v4 = (__int64 *)sub_D960E0(a2);
      v18 = &v4[v5];
      if ( v18 == v4 )
        return 1;
      v6 = 0;
      while ( 1 )
      {
        result = sub_DAD860(a1, *v4, a3);
        if ( !(_DWORD)result )
          break;
        if ( (_DWORD)result == 2 )
          v6 = 1;
        if ( v18 == ++v4 )
        {
          if ( !v6 )
            return 1;
          return 2;
        }
      }
      return result;
    case 8:
      v9 = *(_QWORD *)(a2 + 48);
      if ( v9 == a3 )
        return 2;
      if ( !a3 || (unsigned __int8)sub_B19720(*(_QWORD *)(a1 + 40), **(_QWORD **)(a3 + 32), **(_QWORD **)(v9 + 32)) )
        return 0;
      v10 = *(_QWORD **)(a2 + 48);
      if ( v10 == (_QWORD *)a3 )
        return 1;
      v11 = (_QWORD *)a3;
      break;
    case 0xF:
      v8 = *(_QWORD *)(a2 - 8);
      if ( *(_BYTE *)v8 <= 0x1Cu )
        return 1;
      if ( !a3 )
        return 0;
      v15 = *(_QWORD *)(v8 + 40);
      if ( !*(_BYTE *)(a3 + 84) )
        return sub_C8CA60(a3 + 56, v15) == 0;
      v16 = *(_QWORD **)(a3 + 64);
      v17 = &v16[*(unsigned int *)(a3 + 76)];
      if ( v16 == v17 )
        return 1;
      while ( v15 != *v16 )
      {
        if ( v17 == ++v16 )
          return 1;
      }
      return 0;
    default:
      BUG();
  }
  do
  {
    v11 = (_QWORD *)*v11;
    if ( v10 == v11 )
      return 1;
  }
  while ( v11 );
  v12 = *(__int64 **)(a2 + 32);
  v13 = &v12[*(_QWORD *)(a2 + 40)];
  v14 = v12;
  if ( v13 == v12 )
    return 1;
  while ( sub_DADE90(a1, *v14, a3) )
  {
    if ( v13 == ++v14 )
      return 1;
  }
  return 0;
}
