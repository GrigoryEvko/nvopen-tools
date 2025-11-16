// Function: sub_36FCA50
// Address: 0x36fca50
//
unsigned __int64 __fastcall sub_36FCA50(__int64 a1, __int64 a2)
{
  const char *v2; // r15
  size_t v3; // rdx
  size_t v4; // r14
  int v5; // eax
  __int64 v6; // rdx
  _QWORD *v7; // rcx
  __int64 v8; // rbx
  unsigned __int64 v9; // r12
  __int64 v11; // rax
  unsigned int v12; // r8d
  _QWORD *v13; // rcx
  _QWORD *v14; // rbx
  __int64 *v15; // rax
  __int64 *v16; // rax
  _QWORD *v17; // rax
  unsigned __int64 *v18; // r12
  _QWORD *v19; // [rsp+0h] [rbp-40h]
  unsigned int v20; // [rsp+Ch] [rbp-34h]

  v2 = sub_BD5D20(a2);
  v4 = v3;
  v5 = sub_C92610();
  v6 = (unsigned int)sub_C92740(a1, v2, v4, v5);
  v7 = (_QWORD *)(*(_QWORD *)a1 + 8 * v6);
  v8 = *v7;
  if ( *v7 )
  {
    if ( v8 != -8 )
      goto LABEL_3;
    --*(_DWORD *)(a1 + 16);
  }
  v19 = v7;
  v20 = v6;
  v11 = sub_C7D670(v4 + 17, 8);
  v12 = v20;
  v13 = v19;
  v14 = (_QWORD *)v11;
  if ( v4 )
  {
    memcpy((void *)(v11 + 16), v2, v4);
    v12 = v20;
    v13 = v19;
  }
  *((_BYTE *)v14 + v4 + 16) = 0;
  *v14 = v4;
  v14[1] = 0;
  *v13 = v14;
  ++*(_DWORD *)(a1 + 12);
  v15 = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_C929D0((__int64 *)a1, v12));
  v8 = *v15;
  if ( *v15 == -8 || !v8 )
  {
    v16 = v15 + 1;
    do
    {
      do
        v8 = *v16++;
      while ( !v8 );
    }
    while ( v8 == -8 );
    v9 = *(_QWORD *)(v8 + 8);
    if ( v9 )
      return v9;
    goto LABEL_14;
  }
LABEL_3:
  v9 = *(_QWORD *)(v8 + 8);
  if ( v9 )
    return v9;
LABEL_14:
  v17 = (_QWORD *)sub_22077B0(0x60u);
  if ( v17 )
  {
    memset64(v17, v9, 0xCu);
    *v17 = v17 + 2;
    v17[1] = 0x800000000LL;
  }
  v18 = *(unsigned __int64 **)(v8 + 8);
  *(_QWORD *)(v8 + 8) = v17;
  if ( v18 )
  {
    if ( (unsigned __int64 *)*v18 != v18 + 2 )
      _libc_free(*v18);
    j_j___libc_free_0((unsigned __int64)v18);
  }
  *(_BYTE *)(*(_QWORD *)(v8 + 8) + 88LL) = sub_B91CC0(a2, "thinlto_src_module", 0x12u) != 0;
  return *(_QWORD *)(v8 + 8);
}
