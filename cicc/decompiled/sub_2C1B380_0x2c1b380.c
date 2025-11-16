// Function: sub_2C1B380
// Address: 0x2c1b380
//
bool __fastcall sub_2C1B380(__int64 a1)
{
  __int64 *v1; // rbx
  __int64 v2; // r12
  __int64 *v3; // r13
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 *v6; // r12
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rdx
  _QWORD *v11; // rdi
  __int64 v12; // rax
  bool result; // al
  _QWORD *v14; // rdi
  __int64 v15; // rax
  _QWORD *v16; // rdi
  __int64 v17; // rax
  _QWORD *v18; // rdi
  __int64 v19; // rax

  v1 = *(__int64 **)(a1 + 112);
  v2 = 8LL * *(unsigned int *)(a1 + 120);
  v3 = &v1[(unsigned __int64)v2 / 8];
  v4 = v2 >> 3;
  v5 = v2 >> 5;
  if ( v5 )
  {
    v6 = &v1[4 * v5];
    while ( 1 )
    {
      v10 = *v1;
      if ( *(_BYTE *)(*v1 - 32) == 28 )
      {
        v11 = *(_QWORD **)(v10 + 72);
        v12 = *(unsigned int *)(v10 + 80);
        if ( &v11[v12] != sub_2C0DDE0(v11, (__int64)&v11[v12], v10 - 40) )
          return v3 != v1;
      }
      v7 = v1[1];
      if ( *(_BYTE *)(v7 - 32) == 28 )
      {
        v14 = *(_QWORD **)(v7 + 72);
        v15 = *(unsigned int *)(v7 + 80);
        if ( &v14[v15] != sub_2C0DDE0(v14, (__int64)&v14[v15], v7 - 40) )
          return v3 != v1 + 1;
      }
      v8 = v1[2];
      if ( *(_BYTE *)(v8 - 32) == 28 )
      {
        v16 = *(_QWORD **)(v8 + 72);
        v17 = *(unsigned int *)(v8 + 80);
        if ( &v16[v17] != sub_2C0DDE0(v16, (__int64)&v16[v17], v8 - 40) )
          return v3 != v1 + 2;
      }
      v9 = v1[3];
      if ( *(_BYTE *)(v9 - 32) == 28 )
      {
        v18 = *(_QWORD **)(v9 + 72);
        v19 = *(unsigned int *)(v9 + 80);
        if ( &v18[v19] != sub_2C0DDE0(v18, (__int64)&v18[v19], v9 - 40) )
          return v3 != v1 + 3;
      }
      v1 += 4;
      if ( v6 == v1 )
      {
        v4 = v3 - v1;
        break;
      }
    }
  }
  if ( v4 == 2 )
    goto LABEL_24;
  if ( v4 == 3 )
  {
    if ( sub_2C0DFD0(*v1) )
      return v3 != v1;
    ++v1;
LABEL_24:
    if ( !sub_2C0DFD0(*v1) )
    {
      ++v1;
      goto LABEL_26;
    }
    return v3 != v1;
  }
  if ( v4 != 1 )
    return 0;
LABEL_26:
  result = sub_2C0DFD0(*v1);
  if ( result )
    return v3 != v1;
  return result;
}
