// Function: sub_2FD48B0
// Address: 0x2fd48b0
//
__int64 __fastcall sub_2FD48B0(__int64 a1, __int64 *a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r13d
  __int64 *v5; // rdx
  unsigned int v6; // ebx
  __int64 v7; // rax
  __int64 v8; // rdx
  int v9; // eax
  __int64 *v10; // rdx
  int v11; // r13d
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r9
  __int64 *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // r8
  unsigned __int64 v23; // r10
  __int64 v24; // rdx
  __int64 v25; // [rsp-8h] [rbp-48h]
  __int64 v26; // [rsp+0h] [rbp-40h]
  int v27; // [rsp+0h] [rbp-40h]
  int v28; // [rsp+8h] [rbp-38h]
  unsigned __int64 v29; // [rsp+8h] [rbp-38h]

  v2 = sub_BB98D0((_QWORD *)a1, *a2);
  if ( (_BYTE)v2 )
    return 0;
  v5 = *(__int64 **)(a1 + 8);
  v6 = v2;
  v7 = *v5;
  v8 = v5[1];
  if ( v7 == v8 )
LABEL_32:
    BUG();
  while ( *(_UNKNOWN **)v7 != &unk_501F1C8 )
  {
    v7 += 16;
    if ( v8 == v7 )
      goto LABEL_32;
  }
  v9 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(*(_QWORD *)(v7 + 8), &unk_501F1C8);
  v10 = *(__int64 **)(a1 + 8);
  v11 = v9 + 169;
  v12 = *v10;
  v13 = v10[1];
  if ( v12 == v13 )
LABEL_33:
    BUG();
  while ( *(_UNKNOWN **)v12 != &unk_4F87C64 )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_33;
  }
  v14 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(
                      *(_QWORD *)(v12 + 8),
                      &unk_4F87C64)
                  + 176);
  if ( v14 && *(_QWORD *)(v14 + 8) )
  {
    v15 = *(__int64 **)(a1 + 8);
    v16 = *v15;
    v17 = v15[1];
    if ( v16 == v17 )
LABEL_31:
      BUG();
    while ( *(_UNKNOWN **)v16 != &unk_503BDA8 )
    {
      v16 += 16;
      if ( v17 == v16 )
        goto LABEL_31;
    }
    v28 = v14;
    v18 = a1 + 200;
    v19 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v16 + 8) + 104LL))(
            *(_QWORD *)(v16 + 8),
            &unk_503BDA8);
    v20 = sub_3503E60(v19);
    LODWORD(v14) = v28;
    if ( v20 )
    {
      v26 = v20;
      v21 = sub_22077B0(0x28u);
      LODWORD(v14) = v28;
      LODWORD(v22) = v21;
      if ( v21 )
      {
        *(_QWORD *)(v21 + 8) = 0;
        *(_QWORD *)(v21 + 16) = 0;
        *(_QWORD *)v21 = v26;
        *(_QWORD *)(v21 + 24) = 0;
        *(_DWORD *)(v21 + 32) = 0;
      }
      v23 = *(_QWORD *)(a1 + 376);
      *(_QWORD *)(a1 + 376) = v21;
      if ( v23 )
      {
        v27 = v28;
        v29 = v23;
        sub_C7D6A0(*(_QWORD *)(v23 + 16), 16LL * *(unsigned int *)(v23 + 32), 8);
        j_j___libc_free_0(v29);
        v22 = *(_QWORD *)(a1 + 376);
        LODWORD(v14) = v27;
      }
      goto LABEL_25;
    }
  }
  else
  {
    v18 = a1 + 200;
  }
  LODWORD(v22) = 0;
LABEL_25:
  sub_2FD5DC0(v18, (_DWORD)a2, *(unsigned __int8 *)(a1 + 384), v11, v22, v14, 0, 0);
  v24 = v25;
  do
  {
    v3 = v6;
    v6 = sub_2FDBA40(v18, a2, v24);
  }
  while ( (_BYTE)v6 );
  return v3;
}
