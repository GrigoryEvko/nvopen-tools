// Function: sub_2845C00
// Address: 0x2845c00
//
bool __fastcall sub_2845C00(__int64 *a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rbx
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  __int64 v9; // r13
  __int64 *v10; // rax
  __int64 v11; // r12
  _QWORD *v12; // rax
  _QWORD *v13; // rdx
  __int64 v14; // rax
  bool result; // al
  __int64 v16; // rax
  __int64 v17; // rcx
  int v18; // eax
  int v19; // esi
  unsigned int v20; // edx
  __int64 *v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rax
  int v24; // eax
  int v25; // r8d

  v4 = sub_B46EC0(a2, a3);
  v5 = *a1;
  v6 = v4;
  if ( *(_BYTE *)(*a1 + 1460) )
  {
    v7 = *(_QWORD **)(v5 + 1440);
    v8 = &v7[*(unsigned int *)(v5 + 1452)];
    if ( v7 == v8 )
      return 0;
    while ( v6 != *v7 )
    {
      if ( v8 == ++v7 )
        return 0;
    }
  }
  else if ( !sub_C8CA60(v5 + 1432, v4) )
  {
    return 0;
  }
  v9 = *(_QWORD *)a1[2];
  v10 = (__int64 *)a1[1];
  v11 = *v10;
  if ( *(_BYTE *)(*v10 + 1108) )
  {
    v12 = *(_QWORD **)(v11 + 1088);
    v13 = &v12[*(unsigned int *)(v11 + 1100)];
    if ( v12 == v13 )
      return 0;
    while ( v9 != *v12 )
    {
      if ( v13 == ++v12 )
        return 0;
    }
LABEL_11:
    v14 = sub_2845AB0(v9);
    result = v6 == v14 || v14 == 0;
    if ( result )
      return result;
    v16 = *(_QWORD *)(v11 + 8);
    v17 = *(_QWORD *)(v16 + 8);
    v18 = *(_DWORD *)(v16 + 24);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = (v18 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v21 = (__int64 *)(v17 + 16LL * v20);
      v22 = *v21;
      if ( v9 == *v21 )
      {
LABEL_14:
        v23 = v21[1];
        return *(_QWORD *)v11 != v23;
      }
      v24 = 1;
      while ( v22 != -4096 )
      {
        v25 = v24 + 1;
        v20 = v19 & (v24 + v20);
        v21 = (__int64 *)(v17 + 16LL * v20);
        v22 = *v21;
        if ( v9 == *v21 )
          goto LABEL_14;
        v24 = v25;
      }
    }
    v23 = 0;
    return *(_QWORD *)v11 != v23;
  }
  if ( sub_C8CA60(v11 + 1080, v9) )
    goto LABEL_11;
  return 0;
}
