// Function: sub_922C00
// Address: 0x922c00
//
__int64 __fastcall sub_922C00(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // rbx
  __int64 v6; // r14
  __int64 v7; // rax
  char v8; // di
  __int64 v9; // rdx
  __int64 v10; // r15
  unsigned __int8 v11; // cl
  char v12; // dl
  char v13; // si
  __int64 v14; // rax
  __int64 v15; // rdx
  char v17; // al
  __int64 v18; // rax
  __int64 v19; // [rsp+8h] [rbp-38h]

  v5 = *(_QWORD **)(a3 + 72);
  v19 = v5[2];
  v6 = sub_92F410(a2, v5);
  v7 = *v5;
  v8 = *(_BYTE *)(*v5 + 140LL);
  v9 = *v5;
  if ( v8 == 12 )
  {
    do
      v9 = *(_QWORD *)(v9 + 160);
    while ( *(_BYTE *)(v9 + 140) == 12 );
  }
  v10 = *(_QWORD *)(v9 + 160);
  v11 = 0;
  v12 = *(_BYTE *)(v10 + 140);
  if ( (v12 & 0xFB) == 8 )
  {
    v17 = sub_8D4C10(v10, dword_4F077C4 != 2);
    v12 = *(_BYTE *)(v10 + 140);
    v11 = (v17 & 2) != 0;
    if ( v12 == 12 )
    {
      v18 = v10;
      do
      {
        v18 = *(_QWORD *)(v18 + 160);
        v12 = *(_BYTE *)(v18 + 140);
      }
      while ( v12 == 12 );
    }
    v7 = *v5;
    v8 = *(_BYTE *)(*v5 + 140LL);
  }
  v13 = v12 == 11;
  if ( v8 == 12 )
  {
    do
      v7 = *(_QWORD *)(v7 + 160);
    while ( *(_BYTE *)(v7 + 140) == 12 );
  }
  v14 = *(_QWORD *)(v7 + 160);
  if ( *(_BYTE *)(v14 + 140) == 12 )
  {
    v15 = v14;
    do
      v15 = *(_QWORD *)(v15 + 160);
    while ( *(_BYTE *)(v15 + 140) == 12 );
    if ( *(char *)(v15 + 142) < 0 )
    {
      do
      {
        v14 = *(_QWORD *)(v14 + 160);
        if ( *(_BYTE *)(v14 + 140) != 12 )
          break;
        v14 = *(_QWORD *)(v14 + 160);
      }
      while ( *(_BYTE *)(v14 + 140) == 12 );
    }
    else
    {
      do
        v14 = *(_QWORD *)(v14 + 160);
      while ( *(_BYTE *)(v14 + 140) == 12 );
    }
  }
  sub_9229B0(a1, a2, v6, v10, *(_DWORD *)(v14 + 136), v19, v13, v11);
  return a1;
}
