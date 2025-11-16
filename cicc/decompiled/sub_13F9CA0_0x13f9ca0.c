// Function: sub_13F9CA0
// Address: 0x13f9ca0
//
_QWORD *__fastcall sub_13F9CA0(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax
  __int64 v4; // rbx
  __int64 v5; // rdi
  __int64 v6; // rbx
  unsigned int v7; // r13d
  _QWORD *v8; // r12
  _QWORD *v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  _QWORD *v12; // rdx
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rsi
  _QWORD *v17; // rdx
  _QWORD *v18; // [rsp+0h] [rbp-60h]
  __int64 v19; // [rsp+8h] [rbp-58h]
  _QWORD *v21; // [rsp+18h] [rbp-48h]
  __int64 v22; // [rsp+20h] [rbp-40h]
  int v23; // [rsp+2Ch] [rbp-34h]

  result = *(_QWORD **)(a1 + 40);
  v22 = a1 + 56;
  v18 = result;
  v21 = *(_QWORD **)(a1 + 32);
  if ( v21 == result )
    return result;
  do
  {
    v4 = *v21;
    v19 = *v21;
    v5 = sub_157EBA0(*v21);
    if ( !v5 )
      goto LABEL_23;
    v23 = sub_15F4D60(v5);
    v6 = sub_157EBA0(v4);
    if ( !v23 )
      goto LABEL_23;
    v7 = 0;
    while ( 1 )
    {
      v11 = sub_15F4DF0(v6, v7);
      v12 = *(_QWORD **)(a1 + 72);
      v13 = v11;
      v9 = *(_QWORD **)(a1 + 64);
      if ( v12 != v9 )
        break;
      v8 = &v9[*(unsigned int *)(a1 + 84)];
      if ( v9 == v8 )
      {
        v17 = *(_QWORD **)(a1 + 64);
      }
      else
      {
        do
        {
          if ( v13 == *v9 )
            break;
          ++v9;
        }
        while ( v8 != v9 );
        v17 = v8;
      }
LABEL_18:
      while ( v17 != v9 )
      {
        if ( *v9 < 0xFFFFFFFFFFFFFFFELL )
          goto LABEL_8;
        ++v9;
      }
      if ( v8 == v9 )
        goto LABEL_20;
LABEL_9:
      if ( v23 == ++v7 )
        goto LABEL_23;
    }
    v8 = &v12[*(unsigned int *)(a1 + 80)];
    v9 = (_QWORD *)sub_16CC9F0(v22, v13);
    if ( v13 == *v9 )
    {
      v15 = *(_QWORD *)(a1 + 72);
      if ( v15 == *(_QWORD *)(a1 + 64) )
        v16 = *(unsigned int *)(a1 + 84);
      else
        v16 = *(unsigned int *)(a1 + 80);
      v17 = (_QWORD *)(v15 + 8 * v16);
      goto LABEL_18;
    }
    v10 = *(_QWORD *)(a1 + 72);
    if ( v10 == *(_QWORD *)(a1 + 64) )
    {
      v9 = (_QWORD *)(v10 + 8LL * *(unsigned int *)(a1 + 84));
      v17 = v9;
      goto LABEL_18;
    }
    v9 = (_QWORD *)(v10 + 8LL * *(unsigned int *)(a1 + 80));
LABEL_8:
    if ( v8 != v9 )
      goto LABEL_9;
LABEL_20:
    v14 = *(unsigned int *)(a2 + 8);
    if ( (unsigned int)v14 >= *(_DWORD *)(a2 + 12) )
    {
      sub_16CD150(a2, a2 + 16, 0, 8);
      v14 = *(unsigned int *)(a2 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a2 + 8 * v14) = v19;
    ++*(_DWORD *)(a2 + 8);
LABEL_23:
    result = ++v21;
  }
  while ( v18 != v21 );
  return result;
}
