// Function: sub_385D990
// Address: 0x385d990
//
__int64 *__fastcall sub_385D990(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5, __m128i a6)
{
  __int64 *v8; // r8
  __int64 v9; // rax
  __int64 v10; // rsi
  unsigned int v11; // ecx
  __int64 *v12; // rdx
  __int64 v13; // r9
  __int64 v14; // rax
  __int64 v15; // r15
  __int64 *v16; // rbx
  __int64 v17; // r14
  __int64 v18; // rax
  _QWORD *v19; // rax
  int v21; // edx
  int v22; // r10d

  v8 = sub_1494E70(a1, a3, a5, a6);
  v9 = *(unsigned int *)(a2 + 24);
  if ( !a4 )
    a4 = a3;
  if ( (_DWORD)v9 )
  {
    v10 = *(_QWORD *)(a2 + 8);
    v11 = (v9 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
    v12 = (__int64 *)(v10 + 16LL * v11);
    v13 = *v12;
    if ( a4 == *v12 )
    {
LABEL_5:
      if ( v12 != (__int64 *)(v10 + 16 * v9) )
      {
        v14 = sub_385D970(v12[1]);
        v15 = *(_QWORD *)(a1 + 112);
        v16 = (__int64 *)v14;
        v17 = sub_146F1B0(v15, v14);
        v18 = sub_145CF80(v15, *v16, 1, 0);
        v19 = sub_145DE40(v15, v17, v18);
        sub_1495190(a1, (__int64)v19, a5, a6);
        return sub_1494E70(a1, a3, a5, a6);
      }
    }
    else
    {
      v21 = 1;
      while ( v13 != -8 )
      {
        v22 = v21 + 1;
        v11 = (v9 - 1) & (v21 + v11);
        v12 = (__int64 *)(v10 + 16LL * v11);
        v13 = *v12;
        if ( a4 == *v12 )
          goto LABEL_5;
        v21 = v22;
      }
    }
  }
  return v8;
}
