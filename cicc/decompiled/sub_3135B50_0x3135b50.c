// Function: sub_3135B50
// Address: 0x3135b50
//
__int64 __fastcall sub_3135B50(__int64 a1, void *a2, size_t a3, _DWORD *a4)
{
  int v5; // eax
  unsigned int v6; // edx
  _QWORD *v7; // rcx
  __int64 v8; // rbx
  __int64 result; // rax
  __int64 v10; // rax
  unsigned int v11; // r8d
  _QWORD *v12; // rcx
  _QWORD *v13; // rbx
  __int64 *v14; // rax
  __int64 *v15; // rax
  __int64 v16; // rax
  __int64 v17; // r9
  __int64 v18; // r14
  __int64 v19; // rdx
  bool v20; // al
  __int64 v21; // [rsp+8h] [rbp-78h]
  _QWORD *v22; // [rsp+10h] [rbp-70h]
  unsigned int v23; // [rsp+18h] [rbp-68h]
  __int64 v24; // [rsp+18h] [rbp-68h]
  char v25[32]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v26; // [rsp+40h] [rbp-40h]

  *a4 = a3;
  v5 = sub_C92610();
  v6 = sub_C92740(a1 + 656, a2, a3, v5);
  v7 = (_QWORD *)(*(_QWORD *)(a1 + 656) + 8LL * v6);
  v8 = *v7;
  if ( *v7 )
  {
    if ( v8 != -8 )
      goto LABEL_3;
    --*(_DWORD *)(a1 + 672);
  }
  v22 = v7;
  v23 = v6;
  v10 = sub_C7D670(a3 + 17, 8);
  v11 = v23;
  v12 = v22;
  v13 = (_QWORD *)v10;
  if ( a3 )
  {
    memcpy((void *)(v10 + 16), a2, a3);
    v11 = v23;
    v12 = v22;
  }
  *((_BYTE *)v13 + a3 + 16) = 0;
  *v13 = a3;
  v13[1] = 0;
  *v12 = v13;
  ++*(_DWORD *)(a1 + 668);
  v14 = (__int64 *)(*(_QWORD *)(a1 + 656) + 8LL * (unsigned int)sub_C929D0((__int64 *)(a1 + 656), v11));
  v8 = *v14;
  if ( !*v14 || v8 == -8 )
  {
    v15 = v14 + 1;
    do
    {
      do
        v8 = *v15++;
      while ( v8 == -8 );
    }
    while ( !v8 );
    result = *(_QWORD *)(v8 + 8);
    if ( result )
      return result;
    goto LABEL_14;
  }
LABEL_3:
  result = *(_QWORD *)(v8 + 8);
  if ( result )
    return result;
LABEL_14:
  v16 = sub_AC9B20(**(_QWORD **)(a1 + 504), (char *)a2, a3, 1);
  v17 = *(_QWORD *)(a1 + 504);
  v21 = v16;
  v18 = *(_QWORD *)(v17 + 16);
  v19 = v17 + 8;
  if ( v18 != v17 + 8 )
  {
    do
    {
      if ( !v18 )
        BUG();
      if ( (*(_BYTE *)(v18 + 24) & 1) != 0 )
      {
        v24 = v19;
        v20 = sub_B2FC80(v18 - 56);
        v19 = v24;
        if ( !v20 && v21 == *(_QWORD *)(v18 - 88) )
        {
          result = sub_ADAFB0(v18 - 56, *(_QWORD *)(a1 + 2648));
          *(_QWORD *)(v8 + 8) = result;
          return result;
        }
      }
      v18 = *(_QWORD *)(v18 + 8);
    }
    while ( v19 != v18 );
    v17 = *(_QWORD *)(a1 + 504);
  }
  v26 = 257;
  result = sub_B33830(a1 + 512, (char *)a2, a3, (__int64)v25, 0, v17, 1);
  *(_QWORD *)(v8 + 8) = result;
  return result;
}
