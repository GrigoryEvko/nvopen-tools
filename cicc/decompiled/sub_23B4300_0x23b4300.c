// Function: sub_23B4300
// Address: 0x23b4300
//
__int64 __fastcall sub_23B4300(__int64 a1, __int64 *a2)
{
  const void *v2; // r13
  _QWORD *v4; // rdi
  __int64 v5; // rax
  void *(*v6)(); // rdx
  __int64 v7; // rbx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rax
  void *v12; // rax
  void *(*v13)(); // rax
  void *v14; // rax
  __int64 v15; // r14
  __int64 v16; // r9
  __int64 v17; // rbx
  __int64 v18; // r15
  __int64 v19; // rax
  __int64 v20; // r14
  unsigned __int64 v21; // rdx
  __int64 v22[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = (const void *)(a1 + 16);
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x100000000LL;
  sub_23B2720(v22, a2);
  v4 = (_QWORD *)v22[0];
  if ( !v22[0] )
    goto LABEL_13;
  v5 = *(_QWORD *)v22[0];
  v6 = *(void *(**)())(*(_QWORD *)v22[0] + 24LL);
  if ( v6 == sub_23AE340 )
  {
    if ( &unk_4CDFBF8 == &unk_4C5D161 )
      goto LABEL_4;
LABEL_12:
    (*(void (**)(void))(v5 + 8))();
    goto LABEL_13;
  }
  v12 = v6();
  v4 = (_QWORD *)v22[0];
  if ( v12 != &unk_4C5D161 )
  {
    if ( !v22[0] )
      goto LABEL_13;
    v5 = *(_QWORD *)v22[0];
    goto LABEL_12;
  }
LABEL_4:
  v7 = v4[1];
  (*(void (__fastcall **)(_QWORD *))(*v4 + 8LL))(v4);
  if ( v7 )
  {
    v10 = *(unsigned int *)(a1 + 8);
    if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
    {
      sub_C8D5F0(a1, v2, v10 + 1, 8u, v8, v9);
      v10 = *(unsigned int *)(a1 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a1 + 8 * v10) = v7;
    ++*(_DWORD *)(a1 + 8);
    return a1;
  }
LABEL_13:
  sub_23B2720(v22, a2);
  if ( v22[0]
    && ((v13 = *(void *(**)())(*(_QWORD *)v22[0] + 24LL), v13 != sub_23AE340) ? (v14 = v13()) : (v14 = &unk_4CDFBF8),
        v14 == &unk_4C5D162) )
  {
    v15 = *(_QWORD *)(v22[0] + 8);
    sub_23B42E0(v22);
    if ( v15 )
    {
      v17 = *(_QWORD *)(v15 + 32);
      v18 = v15 + 24;
      if ( v15 + 24 != v17 )
      {
        v19 = *(unsigned int *)(a1 + 8);
        do
        {
          v20 = v17 - 56;
          v21 = v19 + 1;
          if ( !v17 )
            v20 = 0;
          if ( v21 > *(unsigned int *)(a1 + 12) )
          {
            sub_C8D5F0(a1, v2, v21, 8u, 0, v16);
            v19 = *(unsigned int *)(a1 + 8);
          }
          *(_QWORD *)(*(_QWORD *)a1 + 8 * v19) = v20;
          v19 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
          *(_DWORD *)(a1 + 8) = v19;
          v17 = *(_QWORD *)(v17 + 8);
        }
        while ( v18 != v17 );
      }
    }
  }
  else
  {
    sub_23B42E0(v22);
  }
  return a1;
}
