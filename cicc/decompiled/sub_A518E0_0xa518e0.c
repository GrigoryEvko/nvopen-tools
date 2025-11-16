// Function: sub_A518E0
// Address: 0xa518e0
//
_BYTE *__fastcall sub_A518E0(unsigned __int8 *a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // r14d
  unsigned __int8 v6; // bl
  int v7; // eax
  int v8; // ecx
  __int64 v9; // rdi
  __int64 v10; // rax
  _BYTE *result; // rax
  __int64 v12; // rdx
  unsigned int v13; // ebx
  _BYTE *v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rsi
  _BYTE *v17; // rax
  __int64 v18; // rsi
  unsigned int v19; // r14d
  unsigned __int8 v20; // r15
  __int64 v21; // rdx
  int v22; // [rsp+8h] [rbp-38h]

  if ( !a2 )
    return (_BYTE *)sub_904010(a3, "<empty name> ");
  v4 = *a1;
  v6 = *a1;
  v7 = isalpha(v4);
  v8 = a2;
  if ( v7 || (unsigned __int8)(v4 - 36) <= 0x3Bu && (v12 = 0x800000000000601LL, _bittest64(&v12, v4 - 36)) )
  {
    result = *(_BYTE **)(a3 + 32);
    if ( (unsigned __int64)result >= *(_QWORD *)(a3 + 24) )
    {
      result = (_BYTE *)sub_CB5D20(a3, v4);
      v8 = a2;
    }
    else
    {
      *(_QWORD *)(a3 + 32) = result + 1;
      *result = v6;
    }
  }
  else
  {
    v9 = sub_A51310(a3, 0x5Cu);
    v10 = sub_A51310(v9, a0123456789abcd_10[v6 >> 4]);
    result = (_BYTE *)sub_A51310(v10, a0123456789abcd_10[v6 & 0xF]);
    v8 = a2;
  }
  v22 = v8;
  v13 = 1;
  if ( v8 != 1 )
  {
    while ( 1 )
    {
      v19 = a1[v13];
      v20 = a1[v13];
      if ( !isalnum(v19) )
      {
        if ( (unsigned __int8)(v19 - 36) > 0x3Bu )
          break;
        v21 = 0x800000000000601LL;
        if ( !_bittest64(&v21, v19 - 36) )
          break;
      }
      result = *(_BYTE **)(a3 + 32);
      if ( (unsigned __int64)result >= *(_QWORD *)(a3 + 24) )
      {
        result = (_BYTE *)sub_CB5D20(a3, v19);
LABEL_16:
        if ( v22 == ++v13 )
          return result;
      }
      else
      {
        ++v13;
        *(_QWORD *)(a3 + 32) = result + 1;
        *result = v20;
        if ( v22 == v13 )
          return result;
      }
    }
    v14 = *(_BYTE **)(a3 + 32);
    if ( (unsigned __int64)v14 >= *(_QWORD *)(a3 + 24) )
    {
      v15 = sub_CB5D20(a3, 92);
    }
    else
    {
      v15 = a3;
      *(_QWORD *)(a3 + 32) = v14 + 1;
      *v14 = 92;
    }
    v16 = (unsigned __int8)a0123456789abcd_10[v20 >> 4];
    v17 = *(_BYTE **)(v15 + 32);
    if ( (unsigned __int64)v17 >= *(_QWORD *)(v15 + 24) )
    {
      v15 = sub_CB5D20(v15, v16);
    }
    else
    {
      *(_QWORD *)(v15 + 32) = v17 + 1;
      *v17 = v16;
    }
    result = *(_BYTE **)(v15 + 32);
    v18 = (unsigned __int8)a0123456789abcd_10[v20 & 0xF];
    if ( (unsigned __int64)result >= *(_QWORD *)(v15 + 24) )
    {
      result = (_BYTE *)sub_CB5D20(v15, v18);
    }
    else
    {
      *(_QWORD *)(v15 + 32) = result + 1;
      *result = v18;
    }
    goto LABEL_16;
  }
  return result;
}
