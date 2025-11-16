// Function: sub_D34370
// Address: 0xd34370
//
__int64 __fastcall sub_D34370(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // r8
  __int64 v8; // rax
  unsigned int v9; // ecx
  __int64 *v10; // rdx
  __int64 v11; // r9
  __int64 v12; // r15
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  int v18; // edx
  int v19; // r10d

  v5 = sub_DEEF40(a1, a3);
  v6 = *(_QWORD *)(a2 + 8);
  v7 = v5;
  v8 = *(unsigned int *)(a2 + 24);
  if ( (_DWORD)v8 )
  {
    v9 = (v8 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( a3 == *v10 )
    {
LABEL_3:
      if ( v10 != (__int64 *)(v6 + 16 * v8) )
      {
        v12 = v10[1];
        v13 = *(_QWORD *)(a1 + 112);
        v14 = sub_D95540(v12);
        v15 = sub_DA2C50(v13, v14, 1, 0);
        v16 = sub_DA4260(v13, v12, v15);
        sub_DEF380(a1, v16);
        return sub_DEEF40(a1, a3);
      }
    }
    else
    {
      v18 = 1;
      while ( v11 != -4096 )
      {
        v19 = v18 + 1;
        v9 = (v8 - 1) & (v18 + v9);
        v10 = (__int64 *)(v6 + 16LL * v9);
        v11 = *v10;
        if ( a3 == *v10 )
          goto LABEL_3;
        v18 = v19;
      }
    }
  }
  return v7;
}
