// Function: sub_154A520
// Address: 0x154a520
//
_BYTE *__fastcall sub_154A520(char *a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // r14d
  char v6; // bl
  int v7; // r8d
  int v8; // eax
  _BYTE *result; // rax
  unsigned int v10; // ebx
  _BYTE *v11; // rax
  __int64 v12; // rdi
  char v13; // si
  _BYTE *v14; // rax
  char v15; // si
  unsigned int v16; // r14d
  unsigned __int8 v17; // r15
  __int64 v18; // rdx
  __int64 v19; // rdi
  char v20; // al
  unsigned __int8 v21; // si
  __int64 v22; // rdi
  unsigned __int8 v23; // al
  unsigned __int8 v24; // si
  int v25; // [rsp+8h] [rbp-38h]
  int v26; // [rsp+8h] [rbp-38h]

  if ( !a2 )
    return (_BYTE *)sub_1263B40(a3, "<empty name> ");
  v4 = (unsigned __int8)*a1;
  v25 = a2;
  v6 = *a1;
  v8 = isalpha(v4);
  v7 = a2;
  LOBYTE(v8) = v8 != 0;
  if ( (unsigned __int8)(v4 - 36) <= 0x3Bu )
    v8 |= (0x800000000000601uLL >> ((unsigned __int8)v4 - 36)) & 1;
  if ( (_BYTE)v8 )
  {
    result = *(_BYTE **)(a3 + 24);
    if ( (unsigned __int64)result >= *(_QWORD *)(a3 + 16) )
    {
      result = (_BYTE *)sub_16E7DE0(a3, v4);
      v7 = a2;
    }
    else
    {
      *(_QWORD *)(a3 + 24) = result + 1;
      *result = v6;
    }
  }
  else
  {
    v19 = sub_1549FC0(a3, 0x5Cu);
    v20 = *a1 >> 4;
    v21 = v20 + 55;
    if ( (unsigned int)v20 <= 9 )
      v21 = v20 + 48;
    v22 = sub_1549FC0(v19, v21);
    v23 = *a1 & 0xF;
    v24 = v23 + 55;
    if ( v23 <= 9u )
      v24 = v23 + 48;
    result = (_BYTE *)sub_1549FC0(v22, v24);
    v7 = v25;
  }
  v26 = v7;
  v10 = 1;
  if ( v7 != 1 )
  {
    while ( 1 )
    {
      v16 = (unsigned __int8)a1[v10];
      v17 = a1[v10];
      if ( !isalnum(v16) )
      {
        if ( (unsigned __int8)(v16 - 36) > 0x3Bu )
          break;
        v18 = 0x800000000000601LL;
        if ( !_bittest64(&v18, v16 - 36) )
          break;
      }
      result = *(_BYTE **)(a3 + 24);
      if ( (unsigned __int64)result >= *(_QWORD *)(a3 + 16) )
      {
        result = (_BYTE *)sub_16E7DE0(a3, v16);
LABEL_19:
        if ( v26 == ++v10 )
          return result;
      }
      else
      {
        ++v10;
        *(_QWORD *)(a3 + 24) = result + 1;
        *result = v17;
        if ( v26 == v10 )
          return result;
      }
    }
    v11 = *(_BYTE **)(a3 + 24);
    if ( (unsigned __int64)v11 >= *(_QWORD *)(a3 + 16) )
    {
      v12 = sub_16E7DE0(a3, 92);
    }
    else
    {
      v12 = a3;
      *(_QWORD *)(a3 + 24) = v11 + 1;
      *v11 = 92;
    }
    v13 = (v17 >> 4) + 55;
    v14 = *(_BYTE **)(v12 + 24);
    if ( (unsigned __int8)(v17 >> 4) <= 9u )
      v13 = (v17 >> 4) + 48;
    if ( (unsigned __int64)v14 >= *(_QWORD *)(v12 + 16) )
    {
      v12 = sub_16E7DE0(v12, (unsigned int)v13);
    }
    else
    {
      *(_QWORD *)(v12 + 24) = v14 + 1;
      *v14 = v13;
    }
    v15 = (v17 & 0xF) + 55;
    if ( (v17 & 0xFu) <= 9 )
      v15 = (v17 & 0xF) + 48;
    result = *(_BYTE **)(v12 + 24);
    if ( (unsigned __int64)result >= *(_QWORD *)(v12 + 16) )
    {
      result = (_BYTE *)sub_16E7DE0(v12, (unsigned int)v15);
    }
    else
    {
      *(_QWORD *)(v12 + 24) = result + 1;
      *result = v15;
    }
    goto LABEL_19;
  }
  return result;
}
