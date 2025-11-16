// Function: sub_E6DCB0
// Address: 0xe6dcb0
//
__int64 __fastcall sub_E6DCB0(__int64 a1, const void *a2, size_t a3, char a4)
{
  int v5; // eax
  unsigned int v6; // r14d
  _QWORD *v7; // r9
  __int64 v8; // rax
  __int64 v10; // rax
  _QWORD *v11; // r9
  _QWORD *v12; // rcx
  __int64 **v13; // rbx
  __int64 *v14; // r12
  __int64 v15; // rdx
  __int64 v16; // r10
  unsigned __int64 v17; // rax
  _QWORD *v18; // r15
  __int64 v19; // rax
  _QWORD *v20; // [rsp+8h] [rbp-48h]
  _QWORD *v21; // [rsp+10h] [rbp-40h]
  int v22; // [rsp+10h] [rbp-40h]

  v5 = sub_C92610();
  v6 = sub_C92740(a1 + 2216, a2, a3, v5);
  v7 = (_QWORD *)(*(_QWORD *)(a1 + 2216) + 8LL * v6);
  v8 = *v7;
  if ( *v7 )
  {
    if ( v8 != -8 )
      return *(_QWORD *)(v8 + 8);
    --*(_DWORD *)(a1 + 2232);
  }
  v21 = v7;
  v10 = sub_C7D670(a3 + 17, 8);
  v11 = v21;
  v12 = (_QWORD *)v10;
  if ( a3 )
  {
    v20 = (_QWORD *)v10;
    memcpy((void *)(v10 + 16), a2, a3);
    v11 = v21;
    v12 = v20;
  }
  *((_BYTE *)v12 + a3 + 16) = 0;
  *v12 = a3;
  v12[1] = 0;
  *v11 = v12;
  ++*(_DWORD *)(a1 + 2228);
  v13 = (__int64 **)(*(_QWORD *)(a1 + 2216) + 8LL * (unsigned int)sub_C929D0((__int64 *)(a1 + 2216), v6));
  v14 = *v13;
  if ( *v13 )
    goto LABEL_9;
  do
  {
    do
    {
      v14 = v13[1];
      ++v13;
    }
    while ( !v14 );
LABEL_9:
    ;
  }
  while ( v14 == (__int64 *)-8LL );
  v15 = *(_QWORD *)(a1 + 480);
  v16 = *v14;
  *(_QWORD *)(a1 + 560) += 152LL;
  v17 = (v15 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(_QWORD *)(a1 + 488) >= v17 + 152 && v15 )
  {
    *(_QWORD *)(a1 + 480) = v17 + 152;
    v18 = (_QWORD *)((v15 + 7) & 0xFFFFFFFFFFFFFFF8LL);
  }
  else
  {
    v22 = v16;
    v19 = sub_9D1E70(a1 + 480, 152, 152, 3);
    LODWORD(v16) = v22;
    v18 = (_QWORD *)v19;
  }
  sub_E92760((_DWORD)v18, 7, (_DWORD)v14 + 16, v16, (unsigned __int8)(a4 - 2) <= 1u, 0, 0);
  *v18 = &unk_49E35D8;
  (*v13)[1] = (__int64)v18;
  sub_E6B260((_QWORD *)a1, (*v13)[1]);
  return (*v13)[1];
}
