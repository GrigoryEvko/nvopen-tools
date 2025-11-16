// Function: sub_8DBCE0
// Address: 0x8dbce0
//
__int64 __fastcall sub_8DBCE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdx
  __int64 i; // rbx
  char v8; // al
  char v10; // dl
  __int64 v11; // rax
  char v12; // dl
  __int64 v13; // rax
  _QWORD *v14; // rbx
  _QWORD *v15; // r12

  if ( !dword_4D048B8 )
    return 1;
  v6 = *(unsigned __int8 *)(a1 + 140);
  for ( i = a1; (_BYTE)v6 == 12; v6 = *(unsigned __int8 *)(i + 140) )
    i = *(_QWORD *)(i + 160);
  while ( 1 )
  {
    v8 = *(_BYTE *)(a2 + 140);
    if ( v8 != 12 )
      break;
    a2 = *(_QWORD *)(a2 + 160);
  }
  if ( v8 != 7 || (_BYTE)v6 != 7 )
    return 1;
  if ( sub_8DADD0(i, a2, v6, a4, a5) )
    return 0;
  v10 = *(_BYTE *)(i + 140);
  if ( v10 == 12 )
  {
    v11 = i;
    do
    {
      v11 = *(_QWORD *)(v11 + 160);
      v10 = *(_BYTE *)(v11 + 140);
    }
    while ( v10 == 12 );
  }
  if ( !v10 )
    return 1;
  v12 = *(_BYTE *)(a2 + 140);
  if ( v12 == 12 )
  {
    v13 = a2;
    do
    {
      v13 = *(_QWORD *)(v13 + 160);
      v12 = *(_BYTE *)(v13 + 140);
    }
    while ( v12 == 12 );
  }
  if ( !v12 )
    return 1;
  if ( !(unsigned int)sub_8DBAE0(*(_QWORD *)(i + 160), *(_QWORD *)(a2 + 160)) )
    return 0;
  v14 = **(_QWORD ***)(i + 168);
  v15 = **(_QWORD ***)(a2 + 168);
  if ( !v15 || !v14 )
    return 1;
  while ( (unsigned int)sub_8DBAE0(v14[1], v15[1]) )
  {
    v14 = (_QWORD *)*v14;
    v15 = (_QWORD *)*v15;
    if ( !v14 || !v15 )
      return 1;
  }
  return 0;
}
