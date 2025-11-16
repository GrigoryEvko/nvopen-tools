// Function: sub_8E3990
// Address: 0x8e3990
//
void __fastcall sub_8E3990(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v7; // r13
  __int64 v8; // rcx
  __int64 v9; // r14
  int v10; // eax
  __int64 v11; // r15
  __int64 v12; // rdi
  __int64 v13; // rax
  _QWORD *v14; // r12
  _QWORD *v15; // rax
  _QWORD *v16; // rsi
  __int64 v17; // rax
  __int64 v18; // [rsp+8h] [rbp-38h]

  v7 = *(_QWORD *)(a1 + 56);
  v8 = *(_QWORD *)(a1 + 64);
  v9 = *(_QWORD *)(a1 + 48);
  v10 = *(_DWORD *)(a1 + 4);
  if ( v7 <= 1 )
  {
    v14 = (_QWORD *)(a1 + 8);
    if ( v10 && v14 != (_QWORD *)v9 )
    {
      v12 = 16;
      v11 = 2;
      goto LABEL_5;
    }
    v11 = 2;
  }
  else
  {
    a3 = v7 >> 1;
    v11 = v7 + (v7 >> 1) + 1;
    if ( v10 )
    {
      v17 = a1 + 8;
      v12 = 8 * v11;
      if ( v9 != v17 )
        goto LABEL_5;
    }
    if ( v11 > 5 )
    {
      v12 = 8 * v11;
LABEL_5:
      v18 = v8;
      v13 = sub_823970(v12);
      v8 = v18;
      v14 = (_QWORD *)v13;
      goto LABEL_6;
    }
    v14 = (_QWORD *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 4) = 1;
LABEL_6:
  if ( v14 != (_QWORD *)v9 )
  {
    if ( v8 > 0 )
    {
      v15 = v14;
      a3 = v9;
      v16 = &v14[v8];
      do
      {
        if ( v15 )
        {
          v8 = *(_QWORD *)a3;
          *v15 = *(_QWORD *)a3;
        }
        ++v15;
        a3 += 8;
      }
      while ( v16 != v15 );
    }
    if ( v9 == a1 + 8 )
      *(_DWORD *)(a1 + 4) = 0;
    else
      sub_823A00(v9, 8 * v7, a3, v8, a5, a6);
  }
  *(_QWORD *)(a1 + 48) = v14;
  *(_QWORD *)(a1 + 56) = v11;
}
