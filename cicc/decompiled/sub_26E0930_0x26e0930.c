// Function: sub_26E0930
// Address: 0x26e0930
//
__int64 __fastcall sub_26E0930(__int64 a1, __int64 a2)
{
  __int128 v2; // rax
  const void *v3; // r12
  size_t v4; // rdx
  size_t v5; // r14
  int v6; // eax
  unsigned int v7; // r8d
  __int64 *v8; // rcx
  __int64 v9; // rax
  __int64 v11; // rax
  unsigned int v12; // r8d
  __int64 *v13; // rcx
  __int64 v14; // rbx
  __int64 *v15; // rdx
  __int64 *v16; // [rsp+0h] [rbp-40h]
  unsigned int v17; // [rsp+Ch] [rbp-34h]

  *(_QWORD *)&v2 = sub_BD5D20(a2);
  v3 = (const void *)sub_C16140(v2, (__int64)"selected", 8);
  v5 = v4;
  v6 = sub_C92610();
  v7 = sub_C92740(a1 + 96, v3, v5, v6);
  v8 = (__int64 *)(*(_QWORD *)(a1 + 96) + 8LL * v7);
  v9 = *v8;
  if ( *v8 )
  {
    if ( v9 != -8 )
      return v9 + 8;
    --*(_DWORD *)(a1 + 112);
  }
  v16 = v8;
  v17 = v7;
  v11 = sub_C7D670(v5 + 65, 8);
  v12 = v17;
  v13 = v16;
  v14 = v11;
  if ( v5 )
  {
    memcpy((void *)(v11 + 64), v3, v5);
    v12 = v17;
    v13 = v16;
  }
  *(_BYTE *)(v14 + v5 + 64) = 0;
  *(_OWORD *)(v14 + 40) = 0;
  *(_QWORD *)v14 = v5;
  *(_QWORD *)(v14 + 56) = 0;
  *(_QWORD *)(v14 + 8) = v14 + 56;
  *(_QWORD *)(v14 + 16) = 1;
  *(_DWORD *)(v14 + 40) = 1065353216;
  *(_QWORD *)(v14 + 48) = 0;
  *(_OWORD *)(v14 + 24) = 0;
  *v13 = v14;
  ++*(_DWORD *)(a1 + 108);
  v15 = (__int64 *)(*(_QWORD *)(a1 + 96) + 8LL * (unsigned int)sub_C929D0((__int64 *)(a1 + 96), v12));
  v9 = *v15;
  if ( *v15 )
    goto LABEL_9;
  do
  {
    do
    {
      v9 = v15[1];
      ++v15;
    }
    while ( !v9 );
LABEL_9:
    ;
  }
  while ( v9 == -8 );
  return v9 + 8;
}
