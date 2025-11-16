// Function: sub_2A198B0
// Address: 0x2a198b0
//
__int64 __fastcall sub_2A198B0(__int64 **a1, __int64 *a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r12
  __int64 v7; // r15
  __int64 v8; // rdi
  __int64 *v9; // rax
  __int64 result; // rax
  char v11; // dl
  __int64 v12; // rdi
  _QWORD *v13; // rsi
  _QWORD *v14; // rax
  _QWORD *v15; // rdx
  __int64 v16; // rax
  _QWORD *v17; // r13
  _QWORD *v18; // rbx
  _BYTE *v19; // rdx
  __int64 v20; // rax
  __int64 **v21; // rbx
  __int64 v22; // rdi
  _QWORD *v23; // rax
  _QWORD *v24; // rdx
  _QWORD v25[7]; // [rsp+8h] [rbp-38h] BYREF

  v6 = *a1;
  v7 = *a2;
  v8 = **a1;
  if ( !*(_BYTE *)(v8 + 28) )
    goto LABEL_9;
  v9 = *(__int64 **)(v8 + 8);
  a4 = *(unsigned int *)(v8 + 20);
  a3 = &v9[a4];
  if ( v9 != a3 )
  {
    while ( v7 != *v9 )
    {
      if ( a3 == ++v9 )
        goto LABEL_8;
    }
    return 1;
  }
LABEL_8:
  if ( (unsigned int)a4 < *(_DWORD *)(v8 + 16) )
  {
    *(_DWORD *)(v8 + 20) = a4 + 1;
    *a3 = v7;
    ++*(_QWORD *)v8;
    v12 = v6[1];
    v13 = *(_QWORD **)(v7 + 40);
    if ( *(_BYTE *)(v12 + 28) )
    {
LABEL_11:
      v14 = *(_QWORD **)(v12 + 8);
      v15 = &v14[*(unsigned int *)(v12 + 20)];
      if ( v14 == v15 )
        goto LABEL_23;
      while ( v13 != (_QWORD *)*v14 )
      {
        if ( v15 == ++v14 )
          goto LABEL_23;
      }
      goto LABEL_15;
    }
  }
  else
  {
LABEL_9:
    sub_C8CC70(v8, *a2, (__int64)a3, a4, a5, a6);
    if ( !v11 )
      return 1;
    v12 = v6[1];
    v13 = *(_QWORD **)(v7 + 40);
    if ( *(_BYTE *)(v12 + 28) )
      goto LABEL_11;
  }
  if ( !sub_C8CA60(v12, (__int64)v13) )
  {
LABEL_22:
    v13 = *(_QWORD **)(v7 + 40);
LABEL_23:
    v21 = (__int64 **)v6[3];
    v22 = (__int64)*v21;
    if ( *((_BYTE *)*v21 + 28) )
    {
      v23 = *(_QWORD **)(v22 + 8);
      v24 = &v23[*(unsigned int *)(v22 + 20)];
      if ( v23 == v24 )
        return 1;
      while ( (_QWORD *)*v23 != v13 )
      {
        if ( v24 == ++v23 )
          return 1;
      }
    }
    else if ( !sub_C8CA60(v22, (__int64)v13) )
    {
      return 1;
    }
    sub_B444E0((_QWORD *)v7, *v21[1], v21[1][1]);
    return 1;
  }
LABEL_15:
  v16 = 4LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(v7 + 7) & 0x40) != 0 )
  {
    v18 = *(_QWORD **)(v7 - 8);
    v17 = &v18[v16];
  }
  else
  {
    v17 = (_QWORD *)v7;
    v18 = (_QWORD *)(v7 - v16 * 8);
  }
  if ( v17 == v18 )
    goto LABEL_22;
  while ( 1 )
  {
    v19 = (_BYTE *)*v18;
    if ( *(_BYTE *)*v18 > 0x1Cu )
    {
      v20 = v6[2];
      v25[0] = *v18;
      if ( !*(_QWORD *)(v20 + 16) )
        sub_4263D6(v12, v13, v19);
      v13 = v25;
      v12 = v20;
      result = (*(__int64 (__fastcall **)(__int64, _QWORD *))(v20 + 24))(v20, v25);
      if ( !(_BYTE)result )
        return result;
    }
    v18 += 4;
    if ( v17 == v18 )
      goto LABEL_22;
  }
}
