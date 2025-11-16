// Function: sub_70D630
// Address: 0x70d630
//
_QWORD *__fastcall sub_70D630(__int64 a1, __int64 a2, _QWORD *a3, int a4, __int64 a5, _DWORD *a6)
{
  char v7; // r10
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rcx
  __int64 v11; // rcx
  __int64 v12; // rdx
  _QWORD *v13; // rbx
  __int64 v14; // rax
  __int64 v15; // rax
  int v16; // r13d
  _QWORD *result; // rax
  __int64 v18; // rcx
  __int64 v19; // rax
  int v20; // eax

  v7 = *(_BYTE *)(a1 + 192);
  v8 = *(_QWORD *)(a1 + 176);
  if ( a4 )
  {
    v9 = a3[5];
    v10 = 0;
    if ( v8 )
    {
      v10 = *(_QWORD *)(v8 + 104);
      if ( (v7 & 1) != 0 )
        v10 = -v10;
    }
    v11 = v10 - a3[13];
  }
  else
  {
    v9 = a3[7];
    v18 = 0;
    if ( v8 )
    {
      v18 = *(_QWORD *)(v8 + 104);
      if ( (v7 & 1) != 0 )
        v18 = -v18;
    }
    v11 = a3[13] + v18;
  }
  v12 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 200) + 40LL) + 32LL);
  if ( v9 == v12 || v9 && v12 && dword_4F07588 && (v19 = *(_QWORD *)(v9 + 32), *(_QWORD *)(v12 + 32) == v19) && v19 )
  {
    *(_BYTE *)(a1 + 168) &= ~8u;
    v16 = 0;
    v13 = 0;
    *(_QWORD *)(a1 + 128) = a2;
  }
  else
  {
    v13 = **(_QWORD ***)(v9 + 168);
    if ( v13 )
    {
      while ( 1 )
      {
        v15 = v13[5];
        if ( v15 == v12
          || v15 && v12 && dword_4F07588 && (v14 = *(_QWORD *)(v15 + 32), *(_QWORD *)(v12 + 32) == v14) && v14 )
        {
          if ( v13[13] == v11 )
            break;
        }
        v13 = (_QWORD *)*v13;
        if ( !v13 )
          goto LABEL_22;
      }
      v16 = 0;
    }
    else
    {
LABEL_22:
      result = *(_QWORD **)(v12 + 168);
      v13 = (_QWORD *)*result;
      if ( !*result )
      {
LABEL_36:
        *a6 = 1;
        return result;
      }
      v11 = -v11;
      while ( 1 )
      {
        result = (_QWORD *)v13[5];
        if ( result == (_QWORD *)v9
          || result && dword_4F07588 && (result = (_QWORD *)result[4], *(_QWORD **)(v9 + 32) == result) && result )
        {
          if ( v13[13] == v11 )
            break;
        }
        v13 = (_QWORD *)*v13;
        if ( !v13 )
          goto LABEL_36;
      }
      v16 = 1;
    }
    sub_70C9E0(a1, a2, a5, v11, a5);
  }
  v20 = *(unsigned __int8 *)(a1 + 192);
  *(_QWORD *)(a1 + 176) = v13;
  result = (_QWORD *)(v16 | v20 & 0xFFFFFFFE);
  *(_BYTE *)(a1 + 192) = (_BYTE)result;
  return result;
}
