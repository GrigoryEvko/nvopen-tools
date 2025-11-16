// Function: sub_29BEE80
// Address: 0x29bee80
//
__int64 __fastcall sub_29BEE80(__int64 a1, __int64 a2, __int64 a3)
{
  const char *v4; // r15
  size_t v5; // rdx
  size_t v6; // rbx
  int v7; // eax
  unsigned int v8; // r8d
  __int64 *v9; // r9
  __int64 v10; // rdx
  int v11; // ebx
  const char *v12; // r13
  size_t v13; // rdx
  size_t v14; // r15
  int v15; // eax
  int v16; // ebx
  unsigned int v17; // r8d
  __int64 *v18; // r9
  __int64 v19; // rax
  __int64 v21; // rax
  unsigned int v22; // r8d
  __int64 *v23; // r9
  __int64 v24; // rcx
  __int64 *v25; // rdx
  __int64 v26; // rax
  unsigned int v27; // r8d
  __int64 *v28; // r9
  __int64 v29; // rcx
  __int64 *v30; // rax
  __int64 *v31; // rax
  __int64 v32; // [rsp+8h] [rbp-48h]
  __int64 v33; // [rsp+8h] [rbp-48h]
  __int64 *v34; // [rsp+10h] [rbp-40h]
  __int64 *v35; // [rsp+10h] [rbp-40h]
  unsigned int v36; // [rsp+1Ch] [rbp-34h]
  unsigned int v37; // [rsp+1Ch] [rbp-34h]

  v4 = sub_BD5D20(a3);
  v6 = v5;
  v7 = sub_C92610();
  v8 = sub_C92740(a2, v4, v6, v7);
  v9 = (__int64 *)(*(_QWORD *)a2 + 8LL * v8);
  v10 = *v9;
  if ( *v9 )
  {
    if ( v10 != -8 )
      goto LABEL_3;
    --*(_DWORD *)(a2 + 16);
  }
  v35 = v9;
  v37 = v8;
  v26 = sub_C7D670(v6 + 17, 8);
  v27 = v37;
  v28 = v35;
  v29 = v26;
  if ( v6 )
  {
    v32 = v26;
    memcpy((void *)(v26 + 16), v4, v6);
    v27 = v37;
    v28 = v35;
    v29 = v32;
  }
  *(_BYTE *)(v29 + v6 + 16) = 0;
  *(_QWORD *)v29 = v6;
  *(_DWORD *)(v29 + 8) = 0;
  *v28 = v29;
  ++*(_DWORD *)(a2 + 12);
  v30 = (__int64 *)(*(_QWORD *)a2 + 8LL * (unsigned int)sub_C929D0((__int64 *)a2, v27));
  v10 = *v30;
  if ( !*v30 || v10 == -8 )
  {
    v31 = v30 + 1;
    do
    {
      do
        v10 = *v31++;
      while ( v10 == -8 );
    }
    while ( !v10 );
  }
LABEL_3:
  v11 = *(_DWORD *)(v10 + 8);
  v12 = sub_BD5D20(a3);
  v14 = v13;
  v15 = sub_C92610();
  v16 = v11 + 1;
  v17 = sub_C92740(a2, v12, v14, v15);
  v18 = (__int64 *)(*(_QWORD *)a2 + 8LL * v17);
  v19 = *v18;
  if ( *v18 )
  {
    if ( v19 != -8 )
      goto LABEL_5;
    --*(_DWORD *)(a2 + 16);
  }
  v34 = v18;
  v36 = v17;
  v21 = sub_C7D670(v14 + 17, 8);
  v22 = v36;
  v23 = v34;
  v24 = v21;
  if ( v14 )
  {
    v33 = v21;
    memcpy((void *)(v21 + 16), v12, v14);
    v22 = v36;
    v23 = v34;
    v24 = v33;
  }
  *(_BYTE *)(v24 + v14 + 16) = 0;
  *(_QWORD *)v24 = v14;
  *(_DWORD *)(v24 + 8) = 0;
  *v23 = v24;
  ++*(_DWORD *)(a2 + 12);
  v25 = (__int64 *)(*(_QWORD *)a2 + 8LL * (unsigned int)sub_C929D0((__int64 *)a2, v22));
  v19 = *v25;
  if ( *v25 != -8 )
    goto LABEL_11;
  do
  {
    do
    {
      v19 = v25[1];
      ++v25;
    }
    while ( v19 == -8 );
LABEL_11:
    ;
  }
  while ( !v19 );
LABEL_5:
  *(_DWORD *)(v19 + 8) = v16;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
