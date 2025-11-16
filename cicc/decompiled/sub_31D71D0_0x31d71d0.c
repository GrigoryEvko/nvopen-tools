// Function: sub_31D71D0
// Address: 0x31d71d0
//
__int64 __fastcall sub_31D71D0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 result; // rax
  _QWORD **v4; // rbx
  unsigned __int64 v5; // r15
  _QWORD *v8; // r12
  _QWORD *v9; // rax
  int v10; // esi
  unsigned int v11; // esi
  __int64 v12; // rax
  void *v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // rdi
  _BYTE *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rdi
  _QWORD *v20; // rax
  unsigned __int64 v21; // rsi
  __int64 v22; // rdi
  _BYTE *v23; // rax
  _QWORD **v24; // [rsp+8h] [rbp-38h]

  result = *(_QWORD *)(a2 + 16);
  v4 = *(_QWORD ***)(a2 + 8);
  v24 = (_QWORD **)result;
  if ( v4 != (_QWORD **)result )
  {
    v5 = a3;
    do
    {
      v8 = *v4;
      v9 = (_QWORD *)**v4;
      if ( v9 )
      {
        v10 = 1;
        do
        {
          v9 = (_QWORD *)*v9;
          ++v10;
        }
        while ( v9 );
        v11 = 2 * v10;
      }
      else
      {
        v11 = 2;
      }
      v12 = sub_CB69B0(a1, v11);
      v13 = *(void **)(v12 + 32);
      v14 = v12;
      if ( *(_QWORD *)(v12 + 24) - (_QWORD)v13 <= 0xCu )
      {
        v14 = sub_CB6200(v12, "Child Loop BB", 0xDu);
      }
      else
      {
        qmemcpy(v13, "Child Loop BB", 13);
        *(_QWORD *)(v12 + 32) += 13LL;
      }
      v15 = sub_CB59D0(v14, v5);
      v16 = *(_BYTE **)(v15 + 32);
      if ( *(_BYTE **)(v15 + 24) == v16 )
      {
        v15 = sub_CB6200(v15, (unsigned __int8 *)"_", 1u);
      }
      else
      {
        *v16 = 95;
        ++*(_QWORD *)(v15 + 32);
      }
      v17 = sub_CB59F0(v15, *(int *)(*(_QWORD *)v8[4] + 24LL));
      v18 = *(_QWORD *)(v17 + 32);
      v19 = v17;
      if ( (unsigned __int64)(*(_QWORD *)(v17 + 24) - v18) <= 6 )
      {
        v19 = sub_CB6200(v17, " Depth ", 7u);
        v20 = (_QWORD *)*v8;
        if ( !*v8 )
        {
LABEL_21:
          v21 = 1;
          goto LABEL_15;
        }
      }
      else
      {
        *(_DWORD *)v18 = 1885684768;
        *(_WORD *)(v18 + 4) = 26740;
        *(_BYTE *)(v18 + 6) = 32;
        *(_QWORD *)(v17 + 32) += 7LL;
        v20 = (_QWORD *)*v8;
        if ( !*v8 )
          goto LABEL_21;
      }
      LODWORD(v21) = 1;
      do
      {
        v20 = (_QWORD *)*v20;
        v21 = (unsigned int)(v21 + 1);
      }
      while ( v20 );
LABEL_15:
      v22 = sub_CB59D0(v19, v21);
      v23 = *(_BYTE **)(v22 + 32);
      if ( (unsigned __int64)v23 >= *(_QWORD *)(v22 + 24) )
      {
        sub_CB5D20(v22, 10);
      }
      else
      {
        *(_QWORD *)(v22 + 32) = v23 + 1;
        *v23 = 10;
      }
      ++v4;
      result = sub_31D71D0(a1, v8, a3);
    }
    while ( v24 != v4 );
  }
  return result;
}
