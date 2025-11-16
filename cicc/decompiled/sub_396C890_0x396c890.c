// Function: sub_396C890
// Address: 0x396c890
//
__int64 __fastcall sub_396C890(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 result; // rax
  _QWORD **v4; // rbx
  __int64 v5; // r15
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
  __int64 v21; // rsi
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
      v12 = sub_16E8750(a1, v11);
      v13 = *(void **)(v12 + 24);
      v14 = v12;
      if ( *(_QWORD *)(v12 + 16) - (_QWORD)v13 <= 0xCu )
      {
        v14 = sub_16E7EE0(v12, "Child Loop BB", 0xDu);
      }
      else
      {
        qmemcpy(v13, "Child Loop BB", 13);
        *(_QWORD *)(v12 + 24) += 13LL;
      }
      v15 = sub_16E7A90(v14, v5);
      v16 = *(_BYTE **)(v15 + 24);
      if ( *(_BYTE **)(v15 + 16) == v16 )
      {
        v15 = sub_16E7EE0(v15, "_", 1u);
      }
      else
      {
        *v16 = 95;
        ++*(_QWORD *)(v15 + 24);
      }
      v17 = sub_16E7AB0(v15, *(int *)(*(_QWORD *)v8[4] + 48LL));
      v18 = *(_QWORD *)(v17 + 24);
      v19 = v17;
      if ( (unsigned __int64)(*(_QWORD *)(v17 + 16) - v18) <= 6 )
      {
        v19 = sub_16E7EE0(v17, " Depth ", 7u);
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
        *(_QWORD *)(v17 + 24) += 7LL;
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
      v22 = sub_16E7A90(v19, v21);
      v23 = *(_BYTE **)(v22 + 24);
      if ( (unsigned __int64)v23 >= *(_QWORD *)(v22 + 16) )
      {
        sub_16E7DE0(v22, 10);
      }
      else
      {
        *(_QWORD *)(v22 + 24) = v23 + 1;
        *v23 = 10;
      }
      ++v4;
      result = sub_396C890(a1, v8, a3);
    }
    while ( v24 != v4 );
  }
  return result;
}
