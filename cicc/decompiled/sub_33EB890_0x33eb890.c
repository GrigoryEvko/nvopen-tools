// Function: sub_33EB890
// Address: 0x33eb890
//
__int64 __fastcall sub_33EB890(__int64 a1, __int16 *a2)
{
  __int64 result; // rax
  __int64 v3; // r8
  unsigned __int16 v4; // di
  unsigned __int64 v5; // r10
  __int64 v6; // rdx
  unsigned __int64 v7; // rdx
  bool v8; // cf
  bool v9; // zf
  __int64 v10; // rsi
  unsigned __int16 v11; // cx
  bool v12; // r9
  __int64 v13; // rcx
  unsigned __int16 v14; // cx
  bool v15; // si
  __int64 v16; // rcx

  result = *(_QWORD *)(a1 + 16);
  v3 = a1 + 8;
  if ( !result )
    return v3;
  v4 = *a2;
  v5 = *((_QWORD *)a2 + 1);
  while ( 1 )
  {
    v7 = *(_QWORD *)(result + 40);
    v8 = *(_WORD *)(result + 32) < v4;
    v9 = *(_WORD *)(result + 32) == v4;
    if ( *(_WORD *)(result + 32) == v4 )
    {
      v8 = v7 < v5;
      v9 = v7 == v5;
      if ( v7 < v5 )
      {
LABEL_4:
        v6 = *(_QWORD *)(result + 24);
        if ( !v6 )
          return v3;
        goto LABEL_5;
      }
    }
    else if ( *(_WORD *)(result + 32) < v4 )
    {
      goto LABEL_4;
    }
    v6 = *(_QWORD *)(result + 16);
    if ( v8 || v9 )
      break;
    v3 = result;
    if ( !v6 )
      return v3;
LABEL_5:
    result = v6;
  }
  v10 = *(_QWORD *)(result + 24);
  if ( v10 )
  {
    while ( 1 )
    {
      v11 = *(_WORD *)(v10 + 32);
      v12 = v4 < v11;
      if ( v4 == v11 )
        v12 = v5 < *(_QWORD *)(v10 + 40);
      v13 = *(_QWORD *)(v10 + 24);
      if ( v12 )
        v13 = *(_QWORD *)(v10 + 16);
      if ( !v13 )
        break;
      v10 = v13;
    }
  }
  if ( v6 )
  {
    while ( 1 )
    {
      v14 = *(_WORD *)(v6 + 32);
      v15 = v4 > v14;
      if ( v4 == v14 )
        v15 = v5 > *(_QWORD *)(v6 + 40);
      v16 = *(_QWORD *)(v6 + 24);
      if ( !v15 )
      {
        v16 = *(_QWORD *)(v6 + 16);
        result = v6;
      }
      if ( !v16 )
        break;
      v6 = v16;
    }
  }
  return result;
}
