// Function: sub_D86260
// Address: 0xd86260
//
__int64 __fastcall sub_D86260(__int64 a1, __int64 a2)
{
  __int64 v5; // rdi
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // r12
  _BYTE *v9; // rax
  const char *v10; // rax
  size_t v11; // rdx
  _DWORD *v12; // rdi
  unsigned __int8 *v13; // rsi
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  _WORD *v16; // rdx
  __int64 v17; // r12
  _BYTE *v18; // rax
  _WORD *v19; // rdx
  __int64 v21; // rax
  __int64 v22; // rax
  _DWORD *v23; // rdx
  size_t v24; // [rsp+8h] [rbp-28h]

  v5 = a2;
  v6 = a2 + 88;
  sub_ABE8C0(v5, a1);
  v7 = *(_QWORD *)(v6 + 16);
  if ( v7 != v6 )
  {
    while ( 1 )
    {
      v19 = *(_WORD **)(a1 + 32);
      if ( *(_QWORD *)(a1 + 24) - (_QWORD)v19 > 1u )
      {
        v8 = a1;
        *v19 = 8236;
        v9 = (_BYTE *)(*(_QWORD *)(a1 + 32) + 2LL);
        *(_QWORD *)(a1 + 32) = v9;
        if ( *(_BYTE **)(a1 + 24) == v9 )
          goto LABEL_16;
      }
      else
      {
        v8 = sub_CB6200(a1, (unsigned __int8 *)", ", 2u);
        v9 = *(_BYTE **)(v8 + 32);
        if ( *(_BYTE **)(v8 + 24) == v9 )
        {
LABEL_16:
          v8 = sub_CB6200(v8, (unsigned __int8 *)"@", 1u);
          goto LABEL_5;
        }
      }
      *v9 = 64;
      ++*(_QWORD *)(v8 + 32);
LABEL_5:
      v10 = sub_BD5D20(*(_QWORD *)(v7 + 32));
      v12 = *(_DWORD **)(v8 + 32);
      v13 = (unsigned __int8 *)v10;
      v14 = *(_QWORD *)(v8 + 24) - (_QWORD)v12;
      if ( v11 > v14 )
      {
        v21 = sub_CB6200(v8, v13, v11);
        v12 = *(_DWORD **)(v21 + 32);
        v8 = v21;
        v14 = *(_QWORD *)(v21 + 24) - (_QWORD)v12;
      }
      else if ( v11 )
      {
        v24 = v11;
        memcpy(v12, v13, v11);
        v22 = *(_QWORD *)(v8 + 24);
        v23 = (_DWORD *)(*(_QWORD *)(v8 + 32) + v24);
        *(_QWORD *)(v8 + 32) = v23;
        v12 = v23;
        v14 = v22 - (_QWORD)v23;
      }
      if ( v14 <= 3 )
      {
        v8 = sub_CB6200(v8, "(arg", 4u);
      }
      else
      {
        *v12 = 1735549224;
        *(_QWORD *)(v8 + 32) += 4LL;
      }
      v15 = sub_CB59D0(v8, *(_QWORD *)(v7 + 40));
      v16 = *(_WORD **)(v15 + 32);
      v17 = v15;
      if ( *(_QWORD *)(v15 + 24) - (_QWORD)v16 <= 1u )
      {
        v17 = sub_CB6200(v15, (unsigned __int8 *)", ", 2u);
      }
      else
      {
        *v16 = 8236;
        *(_QWORD *)(v15 + 32) += 2LL;
      }
      sub_ABE8C0(v7 + 48, v17);
      v18 = *(_BYTE **)(v17 + 32);
      if ( *(_BYTE **)(v17 + 24) == v18 )
      {
        sub_CB6200(v17, (unsigned __int8 *)")", 1u);
        v7 = sub_220EF30(v7);
        if ( v6 == v7 )
          return a1;
      }
      else
      {
        *v18 = 41;
        ++*(_QWORD *)(v17 + 32);
        v7 = sub_220EF30(v7);
        if ( v6 == v7 )
          return a1;
      }
    }
  }
  return a1;
}
