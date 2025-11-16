// Function: sub_36FFA60
// Address: 0x36ffa60
//
__int64 __fastcall sub_36FFA60(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r13
  __int64 *v4; // rcx
  size_t v5; // rdx
  __int64 *v6; // rdi
  __int64 v7; // rcx
  __int64 v8; // r14
  unsigned __int64 v9; // rbx
  __int64 v10; // rax
  unsigned __int64 v11; // r13
  char v12; // al
  unsigned __int64 v13; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  _WORD *v17; // rdx
  __int64 v18; // r13
  __int64 v19; // rdi
  _BYTE *v20; // rax
  __int64 v21; // [rsp+8h] [rbp-58h] BYREF
  unsigned __int64 v22; // [rsp+10h] [rbp-50h] BYREF
  size_t v23; // [rsp+18h] [rbp-48h]
  __int64 v24; // [rsp+20h] [rbp-40h] BYREF

  v2 = 0;
  sub_3700420(*(_QWORD *)(a1 + 240));
  if ( *(_QWORD *)(a1 + 88) != *(_QWORD *)(a1 + 80) )
  {
    do
    {
      v4 = *(__int64 **)(a1 + 240);
      v5 = *(_QWORD *)(v4[1] + 80 * v2 + 72) * *(_QWORD *)(v4[1] + 80 * v2 + 64);
      a2 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v2++);
      sub_CB6200(*v4, (unsigned __int8 *)a2, v5);
    }
    while ( 0xCCCCCCCCCCCCCCCDLL * ((__int64)(*(_QWORD *)(a1 + 88) - *(_QWORD *)(a1 + 80)) >> 4) > v2 );
  }
  sub_3700640(*(_QWORD *)(a1 + 240));
  v6 = **(__int64 ***)(a1 + 240);
  if ( v6[4] != v6[2] )
    sub_CB5AE0(v6);
  v7 = *(_QWORD *)(a1 + 224);
  v8 = *(_QWORD *)(a1 + 216);
  v9 = v7 - v8;
  if ( v7 != v8 )
  {
    v10 = *(_QWORD *)(a1 + 216);
    v11 = 0;
    while ( 1 )
    {
      a2 = *(unsigned int *)(a1 + 72);
      v6 = (__int64 *)&v22;
      sub_C835A0((__int64)&v22, a2, (void *)(v8 + v11), v7 - v10 - v11);
      v12 = v23;
      LOBYTE(v23) = v23 & 0xFD;
      if ( (v12 & 1) != 0 )
      {
        v13 = v22;
        v22 = 0;
        v21 = v13 | 1;
        if ( (v13 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          sub_C63C30(&v21, a2);
      }
      else
      {
        v11 += v22;
      }
      if ( v9 <= v11 )
        break;
      v7 = *(_QWORD *)(a1 + 224);
      v10 = *(_QWORD *)(a1 + 216);
    }
  }
  if ( (_BYTE)qword_5041608 )
  {
    v15 = sub_C5F790((__int64)v6, a2);
    v16 = sub_CB6200(v15, *(unsigned __int8 **)(a1 + 104), *(_QWORD *)(a1 + 112));
    v17 = *(_WORD **)(v16 + 32);
    v18 = v16;
    if ( *(_QWORD *)(v16 + 24) - (_QWORD)v17 <= 1u )
    {
      v18 = sub_CB6200(v16, (unsigned __int8 *)": ", 2u);
    }
    else
    {
      *v17 = 8250;
      *(_QWORD *)(v16 + 32) += 2LL;
    }
    sub_310D670((__int64)&v22, *(int **)(a1 + 216), a1 + 104);
    v19 = sub_CB6200(v18, (unsigned __int8 *)v22, v23);
    v20 = *(_BYTE **)(v19 + 32);
    if ( *(_BYTE **)(v19 + 24) == v20 )
    {
      sub_CB6200(v19, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v20 = 10;
      ++*(_QWORD *)(v19 + 32);
    }
    if ( (__int64 *)v22 != &v24 )
      j_j___libc_free_0(v22);
  }
  return *(_QWORD *)(a1 + 216);
}
