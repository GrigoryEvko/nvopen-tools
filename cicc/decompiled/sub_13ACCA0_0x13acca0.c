// Function: sub_13ACCA0
// Address: 0x13acca0
//
__int64 __fastcall sub_13ACCA0(__int64 a1, __int64 a2, char a3, _QWORD *a4)
{
  _QWORD *v4; // r15
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // rax
  unsigned int v9; // ecx
  unsigned int v10; // edx
  __int64 i; // r13
  __int64 v12; // rax
  __int64 *v13; // rax
  _QWORD **v14; // r12
  __int64 v15; // r14
  __int64 *v16; // r14
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rax

  v4 = (_QWORD *)a2;
  v5 = *(_QWORD *)(a1 + 8);
  v6 = sub_1456040(a2);
  v7 = sub_145CF80(v5, v6, 0, 0);
  v8 = sub_2207820(32LL * (unsigned int)(*(_DWORD *)(a1 + 40) + 1));
  v9 = *(_DWORD *)(a1 + 40);
  v10 = 1;
  for ( i = v8; v10 <= v9; v13[3] = 0 )
  {
    v12 = v10++;
    v13 = (__int64 *)(i + 32 * v12);
    *v13 = v7;
    v13[1] = v7;
    v13[2] = v7;
  }
  if ( *(_WORD *)(a2 + 24) == 7 )
  {
    do
    {
      v14 = (_QWORD **)v4[6];
      if ( a3 )
        v15 = (unsigned int)sub_13A6B70(a1, v14);
      else
        v15 = (unsigned int)sub_13A6B90(a1, v14);
      v16 = (__int64 *)(i + 32 * v15);
      v17 = sub_13A5BC0(v4, *(_QWORD *)(a1 + 8));
      *v16 = v17;
      v18 = sub_13AC6E0(a1, v17);
      v19 = *v16;
      v16[1] = v18;
      v16[2] = sub_13AC720(a1, v19);
      v20 = sub_1456040(v4);
      v16[3] = sub_13A7AF0(a1, (__int64)v14, v20);
      v21 = v4[4];
      v4 = *(_QWORD **)v21;
    }
    while ( *(_WORD *)(*(_QWORD *)v21 + 24LL) == 7 );
  }
  *a4 = v4;
  return i;
}
