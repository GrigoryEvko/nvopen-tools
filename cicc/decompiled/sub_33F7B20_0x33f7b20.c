// Function: sub_33F7B20
// Address: 0x33f7b20
//
__int64 __fastcall sub_33F7B20(__int64 a1, __int16 *a2)
{
  __int64 v4; // rbx
  unsigned __int16 v5; // si
  unsigned __int64 v6; // r9
  unsigned __int16 v7; // dx
  unsigned __int64 v8; // rdi
  bool v9; // cl
  __int64 v10; // rax
  bool v11; // cc
  __int64 v13; // rax
  unsigned __int16 v14; // dx

  v4 = *(_QWORD *)(a1 + 16);
  if ( v4 )
  {
    v5 = *a2;
    v6 = *((_QWORD *)a2 + 1);
    while ( 1 )
    {
      v7 = *(_WORD *)(v4 + 32);
      v8 = *(_QWORD *)(v4 + 40);
      v9 = v5 < v7;
      if ( v5 == v7 )
        v9 = v8 > v6;
      v10 = *(_QWORD *)(v4 + 24);
      if ( v9 )
        v10 = *(_QWORD *)(v4 + 16);
      if ( !v10 )
        break;
      v4 = v10;
    }
    if ( !v9 )
    {
      v11 = v5 <= v7;
      if ( v5 != v7 )
        goto LABEL_11;
      goto LABEL_16;
    }
  }
  else
  {
    v4 = a1 + 8;
  }
  if ( v4 != *(_QWORD *)(a1 + 24) )
  {
    v13 = sub_220EF80(v4);
    v6 = *((_QWORD *)a2 + 1);
    v14 = *(_WORD *)(v13 + 32);
    v8 = *(_QWORD *)(v13 + 40);
    v4 = v13;
    v11 = (unsigned __int16)*a2 <= v14;
    if ( *a2 != v14 )
    {
LABEL_11:
      if ( v11 )
        return v4;
      return 0;
    }
LABEL_16:
    if ( v8 >= v6 )
      return v4;
    return 0;
  }
  return 0;
}
