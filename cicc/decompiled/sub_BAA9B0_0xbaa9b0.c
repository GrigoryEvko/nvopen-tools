// Function: sub_BAA9B0
// Address: 0xbaa9b0
//
_BYTE *__fastcall sub_BAA9B0(__int64 a1, __int64 a2, char a3)
{
  const char *v5; // rsi
  unsigned __int64 v6; // rdx
  _BYTE *v7; // rax
  _BYTE *v8; // r14
  _QWORD *v9; // r13
  __int64 v10; // rax
  _QWORD *v11; // r12
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v15; // [rsp+8h] [rbp-38h]

  v5 = "llvm.compiler.used";
  v6 = (-(__int64)(a3 == 0) & 0xFFFFFFFFFFFFFFF7LL) + 18;
  if ( !a3 )
    v5 = "llvm.used";
  v7 = sub_BA8CD0(a1, (__int64)v5, v6, 0);
  v8 = v7;
  if ( v7 && !sub_B2FC80((__int64)v7) )
  {
    v9 = (_QWORD *)*((_QWORD *)v8 - 4);
    v10 = 32LL * (*((_DWORD *)v9 + 1) & 0x7FFFFFF);
    if ( (*((_BYTE *)v9 + 7) & 0x40) != 0 )
    {
      v11 = (_QWORD *)*(v9 - 1);
      v9 = &v11[(unsigned __int64)v10 / 8];
    }
    else
    {
      v11 = &v9[v10 / 0xFFFFFFFFFFFFFFF8LL];
    }
    for ( ; v9 != v11; ++*(_DWORD *)(a2 + 8) )
    {
      v12 = sub_BD3990(*v11);
      v13 = *(unsigned int *)(a2 + 8);
      if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        v15 = v12;
        sub_C8D5F0(a2, a2 + 16, v13 + 1, 8);
        v13 = *(unsigned int *)(a2 + 8);
        v12 = v15;
      }
      v11 += 4;
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v13) = v12;
    }
  }
  return v8;
}
