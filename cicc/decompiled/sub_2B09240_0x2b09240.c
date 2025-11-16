// Function: sub_2B09240
// Address: 0x2b09240
//
__int64 __fastcall sub_2B09240(_BYTE *a1, int a2, __int64 a3, int a4)
{
  char v4; // al
  _BYTE *v7; // r14
  char v8; // bl
  __int64 v10; // rdi
  __int64 v11; // rdi
  unsigned __int64 v12; // rax
  char *v13; // rdx
  __int64 v14; // rdx
  _BYTE *v15; // rax
  __int64 v16; // rax
  unsigned int v17; // r13d
  _QWORD *v18; // rax
  int v19; // [rsp+8h] [rbp-38h]
  int v20; // [rsp+8h] [rbp-38h]

  v4 = *a1;
  if ( *a1 == 5 )
    return 0;
  v7 = a1;
  v8 = 0;
  while ( 1 )
  {
    if ( v4 != 58 )
    {
      if ( v4 != 54 )
        goto LABEL_5;
      v10 = *((_QWORD *)v7 - 4);
      if ( *(_BYTE *)v10 == 17 )
      {
        v11 = v10 + 24;
      }
      else
      {
        v14 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v10 + 8) + 8LL) - 17;
        if ( (unsigned int)v14 > 1 )
          goto LABEL_5;
        if ( *(_BYTE *)v10 > 0x15u )
          goto LABEL_5;
        v20 = a4;
        v15 = sub_AD7630(v10, 0, v14);
        a4 = v20;
        if ( !v15 || *v15 != 17 )
          goto LABEL_5;
        v11 = (__int64)(v15 + 24);
      }
      v19 = a4;
      v12 = sub_C459C0(v11, 8u);
      a4 = v19;
      if ( v12 )
        goto LABEL_5;
    }
    v13 = (char *)*((_QWORD *)v7 - 8);
    if ( *v7 == 58 )
      v8 = 1;
    v4 = *v13;
    if ( *v13 == 5 )
      break;
    v7 = (_BYTE *)*((_QWORD *)v7 - 8);
  }
  v7 = (_BYTE *)*((_QWORD *)v7 - 8);
LABEL_5:
  if ( ((unsigned __int8)a4 & ((unsigned __int8)v8 ^ 1)) != 0 )
    return 0;
  if ( a1 == v7 )
    return 0;
  if ( *v7 != 68 )
    return 0;
  v16 = *((_QWORD *)v7 - 4);
  if ( !v16 || *(_BYTE *)v16 != 61 )
    return 0;
  v17 = (*(_DWORD *)(*(_QWORD *)(v16 + 8) + 8LL) >> 8) * a2;
  v18 = (_QWORD *)sub_BD5C60((__int64)a1);
  sub_BCCE00(v18, v17);
  return sub_DFA8F0(a3);
}
