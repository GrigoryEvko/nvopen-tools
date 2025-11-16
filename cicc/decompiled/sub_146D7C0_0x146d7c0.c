// Function: sub_146D7C0
// Address: 0x146d7c0
//
__int64 __fastcall sub_146D7C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v6; // rbx
  char v7; // r12
  int v8; // eax
  __int64 v10; // r13
  int v11; // ebx
  int v12; // eax
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 *v15; // [rsp+8h] [rbp-38h]

  switch ( *(_WORD *)(a2 + 24) )
  {
    case 0:
      return 2;
    case 1:
    case 2:
    case 3:
      return sub_146D410(a1, *(_QWORD *)(a2 + 32), a3);
    case 4:
    case 5:
    case 8:
    case 9:
      goto LABEL_3;
    case 6:
      v10 = *(_QWORD *)(a2 + 40);
      v11 = sub_146D410(a1, *(_QWORD *)(a2 + 32), a3);
      if ( !v11 )
        return 0;
      v12 = sub_146D410(a1, v10, a3);
      if ( !v12 )
        return 0;
      if ( v11 != 2 || v12 != 2 )
        return 1;
      return 2;
    case 7:
      if ( !(unsigned __int8)sub_15CC8F0(*(_QWORD *)(a1 + 56), **(_QWORD **)(*(_QWORD *)(a2 + 48) + 32LL), a3, a4, a5) )
        return 0;
LABEL_3:
      v6 = *(__int64 **)(a2 + 32);
      v15 = &v6[*(_QWORD *)(a2 + 40)];
      if ( v15 == v6 )
        return 2;
      v7 = 1;
      break;
    case 0xA:
      v13 = *(_QWORD *)(a2 - 8);
      if ( *(_BYTE *)(v13 + 16) <= 0x17u )
        return 2;
      v14 = *(_QWORD *)(v13 + 40);
      if ( a3 == v14 )
        return 1;
      if ( (unsigned __int8)sub_15CC890(*(_QWORD *)(a1 + 56), v14, a3, a4, a5) )
        return 2;
      return 0;
  }
  do
  {
    v8 = sub_146D410(a1, *v6, a3);
    if ( !v8 )
      return 0;
    if ( v8 == 1 )
      v7 = 0;
    ++v6;
  }
  while ( v15 != v6 );
  if ( v7 )
    return 2;
  else
    return 1;
}
