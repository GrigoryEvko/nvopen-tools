// Function: sub_E539A0
// Address: 0xe539a0
//
_BYTE *__fastcall sub_E539A0(
        __int64 a1,
        unsigned int a2,
        unsigned int a3,
        unsigned int a4,
        unsigned int a5,
        __int64 a6,
        char a7)
{
  unsigned int v7; // r9d
  __int64 v12; // r12
  _BYTE *v13; // rax
  char *v14; // rsi
  size_t v15; // rdx
  _QWORD *v16; // rax
  _BYTE *v17; // rdx
  __int64 v18; // rax
  _WORD *v19; // rdx
  __int64 v20; // rdi
  unsigned __int64 v22; // r8
  char *v23; // rax
  char *v24; // rsi
  unsigned int v25; // eax
  unsigned int v26; // eax
  unsigned int v27; // ecx
  __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // rax

  v7 = a2;
  v12 = *(_QWORD *)(a1 + 304);
  v13 = *(_BYTE **)(v12 + 32);
  if ( (unsigned __int64)v13 >= *(_QWORD *)(v12 + 24) )
  {
    v29 = sub_CB5D20(v12, 9);
    v7 = a2;
    v12 = v29;
  }
  else
  {
    *(_QWORD *)(v12 + 32) = v13 + 1;
    *v13 = 9;
  }
  if ( v7 == 1 )
  {
    v14 = ".macosx_version_min";
  }
  else if ( v7 <= 1 )
  {
    v14 = ".ios_version_min";
  }
  else if ( v7 == 2 )
  {
    v14 = ".tvos_version_min";
  }
  else
  {
    if ( v7 != 3 )
      BUG();
    v14 = ".watchos_version_min";
  }
  v15 = strlen(v14);
  v16 = *(_QWORD **)(v12 + 32);
  if ( *(_QWORD *)(v12 + 24) - (_QWORD)v16 >= v15 )
  {
    v22 = (unsigned __int64)(v16 + 1) & 0xFFFFFFFFFFFFFFF8LL;
    *v16 = *(_QWORD *)v14;
    *(_QWORD *)((char *)v16 + (unsigned int)v15 - 8) = *(_QWORD *)&v14[(unsigned int)v15 - 8];
    v23 = (char *)v16 - v22;
    v24 = (char *)(v14 - v23);
    v25 = (v15 + (_DWORD)v23) & 0xFFFFFFF8;
    if ( v25 >= 8 )
    {
      v26 = v25 & 0xFFFFFFF8;
      v27 = 0;
      do
      {
        v28 = v27;
        v27 += 8;
        *(_QWORD *)(v22 + v28) = *(_QWORD *)&v24[v28];
      }
      while ( v27 < v26 );
    }
    v17 = (_BYTE *)(*(_QWORD *)(v12 + 32) + v15);
    *(_QWORD *)(v12 + 32) = v17;
    if ( *(_QWORD *)(v12 + 24) > (unsigned __int64)v17 )
      goto LABEL_9;
  }
  else
  {
    v12 = sub_CB6200(v12, (unsigned __int8 *)v14, v15);
    v17 = *(_BYTE **)(v12 + 32);
    if ( *(_QWORD *)(v12 + 24) > (unsigned __int64)v17 )
    {
LABEL_9:
      *(_QWORD *)(v12 + 32) = v17 + 1;
      *v17 = 32;
      goto LABEL_10;
    }
  }
  v12 = sub_CB5D20(v12, 32);
LABEL_10:
  v18 = sub_CB59D0(v12, a3);
  v19 = *(_WORD **)(v18 + 32);
  v20 = v18;
  if ( *(_QWORD *)(v18 + 24) - (_QWORD)v19 <= 1u )
  {
    v20 = sub_CB6200(v18, (unsigned __int8 *)", ", 2u);
  }
  else
  {
    *v19 = 8236;
    *(_QWORD *)(v18 + 32) += 2LL;
  }
  sub_CB59D0(v20, a4);
  if ( a5 )
  {
    v30 = sub_904010(*(_QWORD *)(a1 + 304), ", ");
    sub_CB59D0(v30, a5);
  }
  sub_E534F0(*(_QWORD *)(a1 + 304), &a7);
  return sub_E4D880(a1);
}
