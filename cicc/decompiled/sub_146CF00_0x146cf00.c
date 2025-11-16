// Function: sub_146CF00
// Address: 0x146cf00
//
__int64 __fastcall sub_146CF00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v7; // rbx
  char v8; // r12
  int v9; // eax
  __int64 v10; // rax
  int v11; // ebx
  int v12; // eax
  __int64 v13; // rax
  _QWORD *v14; // rdx
  _QWORD *v15; // rax
  __int64 *v16; // rbx
  __int64 *v17; // r12
  __int64 *v18; // [rsp+8h] [rbp-38h]

  switch ( *(_WORD *)(a2 + 24) )
  {
    case 0:
      return 1;
    case 1:
    case 2:
    case 3:
      return sub_146CB30(a1, *(_QWORD *)(a2 + 32), a3);
    case 4:
    case 5:
    case 8:
    case 9:
      v7 = *(__int64 **)(a2 + 32);
      v18 = &v7[*(_QWORD *)(a2 + 40)];
      if ( v7 == v18 )
        return 1;
      v8 = 0;
      do
      {
        v9 = sub_146CB30(a1, *v7, a3);
        if ( !v9 )
          return 0;
        if ( v9 == 2 )
          v8 = 1;
        ++v7;
      }
      while ( v18 != v7 );
      if ( !v8 )
        return 1;
      return 2;
    case 6:
      v11 = sub_146CB30(a1, *(_QWORD *)(a2 + 32), a3);
      if ( !v11 )
        return 0;
      v12 = sub_146CB30(a1, *(_QWORD *)(a2 + 40), a3);
      if ( !v12 )
        return 0;
      if ( v11 != 1 || v12 != 1 )
        return 2;
      return 1;
    case 7:
      v13 = *(_QWORD *)(a2 + 48);
      if ( a3 == v13 )
        return 2;
      if ( !a3
        || (unsigned __int8)sub_15CC8F0(*(_QWORD *)(a1 + 56), **(_QWORD **)(a3 + 32), **(_QWORD **)(v13 + 32), a4, a5) )
      {
        return 0;
      }
      v14 = *(_QWORD **)(a2 + 48);
      if ( v14 == (_QWORD *)a3 )
        return 1;
      v15 = (_QWORD *)a3;
      break;
    case 0xA:
      v10 = *(_QWORD *)(a2 - 8);
      if ( *(_BYTE *)(v10 + 16) <= 0x17u )
        return 1;
      if ( !a3 )
        return 0;
      return !sub_1377F70(a3 + 56, *(_QWORD *)(v10 + 40));
  }
  do
  {
    v15 = (_QWORD *)*v15;
    if ( v14 == v15 )
      return 1;
  }
  while ( v15 );
  v16 = *(__int64 **)(a2 + 32);
  v17 = &v16[*(_QWORD *)(a2 + 40)];
  if ( v16 == v17 )
    return 1;
  while ( sub_146CEE0(a1, *v16, a3) )
  {
    if ( v17 == ++v16 )
      return 1;
  }
  return 0;
}
