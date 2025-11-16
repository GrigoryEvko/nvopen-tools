// Function: sub_3962830
// Address: 0x3962830
//
_QWORD *__fastcall sub_3962830(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *result; // rax
  char v6; // dl
  char v7; // r13
  __int64 v8; // rax
  __int64 *v9; // rbx
  __int64 *v10; // r15
  _QWORD *v11; // r12
  _QWORD *v12; // rax
  __int64 v13; // rax
  __int64 v14; // rsi
  _QWORD *v15; // rdx
  __int64 *v16; // r15
  __int64 v17; // rax
  __int64 *v18; // r12
  __int64 v19; // r13
  __int64 v20; // rdx
  __int64 v21; // rsi
  _QWORD *v22; // rdx
  __int64 v25; // [rsp+10h] [rbp-40h]
  __int64 v26[7]; // [rsp+18h] [rbp-38h] BYREF

  v26[0] = a1;
  result = sub_1412190(a4, a1);
  if ( !v6 )
    return result;
  v7 = v6;
  v8 = 3LL * (*(_DWORD *)(v26[0] + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(v26[0] + 23) & 0x40) != 0 )
  {
    v10 = *(__int64 **)(v26[0] - 8);
    v9 = &v10[v8];
  }
  else
  {
    v9 = (__int64 *)v26[0];
    v10 = (__int64 *)(v26[0] - v8 * 8);
  }
  if ( v10 == v9 )
    return (_QWORD *)sub_14EF3D0(a3, v26);
  do
  {
    v14 = *v10;
    if ( *(_BYTE *)(*v10 + 16) <= 0x17u )
      goto LABEL_11;
    v15 = *(_QWORD **)(a2 + 16);
    v12 = *(_QWORD **)(a2 + 8);
    if ( v15 == v12 )
    {
      v11 = &v12[*(unsigned int *)(a2 + 28)];
      if ( v12 == v11 )
      {
        v22 = *(_QWORD **)(a2 + 8);
      }
      else
      {
        do
        {
          if ( v14 == *v12 )
            break;
          ++v12;
        }
        while ( v11 != v12 );
        v22 = v11;
      }
    }
    else
    {
      v25 = *v10;
      v11 = &v15[*(unsigned int *)(a2 + 24)];
      v12 = sub_16CC9F0(a2, v14);
      if ( v25 == *v12 )
      {
        v20 = *(_QWORD *)(a2 + 16);
        if ( v20 == *(_QWORD *)(a2 + 8) )
          v21 = *(unsigned int *)(a2 + 28);
        else
          v21 = *(unsigned int *)(a2 + 24);
        v22 = (_QWORD *)(v20 + 8 * v21);
      }
      else
      {
        v13 = *(_QWORD *)(a2 + 16);
        if ( v13 != *(_QWORD *)(a2 + 8) )
        {
          v12 = (_QWORD *)(v13 + 8LL * *(unsigned int *)(a2 + 24));
          goto LABEL_9;
        }
        v12 = (_QWORD *)(v13 + 8LL * *(unsigned int *)(a2 + 28));
        v22 = v12;
      }
    }
    while ( v22 != v12 && *v12 >= 0xFFFFFFFFFFFFFFFELL )
      ++v12;
LABEL_9:
    if ( v12 != v11 )
      v7 = 0;
LABEL_11:
    v10 += 3;
  }
  while ( v9 != v10 );
  if ( !v7 )
  {
    v16 = (__int64 *)v26[0];
    v17 = 3LL * (*(_DWORD *)(v26[0] + 20) & 0xFFFFFFF);
    if ( (*(_BYTE *)(v26[0] + 23) & 0x40) != 0 )
    {
      v18 = *(__int64 **)(v26[0] - 8);
      v16 = &v18[v17];
    }
    else
    {
      v18 = (__int64 *)(v26[0] - v17 * 8);
    }
    for ( ; v16 != v18; v18 += 3 )
    {
      v19 = *v18;
      if ( *(_BYTE *)(*v18 + 16) > 0x17u && sub_13A0E30(a2, *v18) && !sub_13A0E30(a4, v19) )
        sub_3962830(v19, a2, a3, a4);
    }
  }
  return (_QWORD *)sub_14EF3D0(a3, v26);
}
