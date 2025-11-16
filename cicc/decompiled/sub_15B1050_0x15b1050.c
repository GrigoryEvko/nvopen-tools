// Function: sub_15B1050
// Address: 0x15b1050
//
bool __fastcall sub_15B1050(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rdi
  __int64 v5; // rax
  size_t v6; // rdx
  size_t v7; // r13
  const void *v8; // r14
  const void *v9; // rax
  __int64 v10; // rdx
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rbx
  size_t v16; // rdx
  __int64 v17; // rdx

  if ( a1 == sub_1626D20(a2) )
    return 1;
  v3 = *(unsigned int *)(a1 + 8);
  v4 = *(_QWORD *)(a1 + 8 * (3 - v3));
  if ( !v4 )
  {
LABEL_7:
    v12 = *(_QWORD *)(a1 + 8 * (2 - v3));
    if ( v12 )
    {
      v13 = sub_161E970(v12);
      v15 = v14;
      v8 = (const void *)v13;
      v9 = (const void *)sub_1649960(a2);
      v7 = v16;
      if ( v15 != v16 )
        return 0;
      if ( v16 )
        return memcmp(v9, v8, v7) == 0;
    }
    else
    {
      sub_1649960(a2);
      if ( v17 )
        return 0;
    }
    return 1;
  }
  v5 = sub_161E970(v4);
  v7 = v6;
  if ( !v6 )
  {
    v3 = *(unsigned int *)(a1 + 8);
    goto LABEL_7;
  }
  v8 = (const void *)v5;
  v9 = (const void *)sub_1649960(a2);
  if ( v7 != v10 )
    return 0;
  return memcmp(v9, v8, v7) == 0;
}
