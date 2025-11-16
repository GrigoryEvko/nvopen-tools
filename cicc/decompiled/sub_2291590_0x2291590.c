// Function: sub_2291590
// Address: 0x2291590
//
__int64 __fastcall sub_2291590(__int64 a1, __int64 a2, char a3, _QWORD *a4)
{
  _QWORD *v4; // r15
  __int64 v5; // r14
  __int64 v6; // rax
  _QWORD *v7; // r14
  __int64 v8; // rax
  unsigned int v9; // ecx
  unsigned int v10; // edx
  __int64 i; // r13
  __int64 v12; // rax
  _QWORD *v13; // rax
  char *v14; // r12
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r14
  __int64 *v19; // r14
  __int64 v20; // rax
  unsigned __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rax

  v4 = (_QWORD *)a2;
  v5 = *(_QWORD *)(a1 + 8);
  v6 = sub_D95540(a2);
  v7 = sub_DA2C50(v5, v6, 0, 0);
  v8 = sub_2207820(32LL * (unsigned int)(*(_DWORD *)(a1 + 40) + 1));
  v9 = *(_DWORD *)(a1 + 40);
  v10 = 1;
  for ( i = v8; v10 <= v9; v13[3] = 0 )
  {
    v12 = v10++;
    v13 = (_QWORD *)(i + 32 * v12);
    *v13 = v7;
    v13[1] = v7;
    v13[2] = v7;
  }
  if ( *(_WORD *)(a2 + 24) == 8 )
  {
    do
    {
      v14 = (char *)v4[6];
      if ( a3 )
        v18 = (unsigned int)sub_228D710(a1, (_QWORD **)v14);
      else
        v18 = (unsigned int)sub_228D730(a1, (_QWORD **)v14);
      v19 = (__int64 *)(i + 32 * v18);
      v20 = sub_D33D80(v4, *(_QWORD *)(a1 + 8), v15, v16, v17);
      *v19 = v20;
      v21 = sub_2290F30(a1, v20);
      v22 = *v19;
      v19[1] = v21;
      v19[2] = sub_2290F70(a1, v22);
      v23 = sub_D95540((__int64)v4);
      v19[3] = (__int64)sub_228E360(a1, v14, v23);
      v24 = v4[4];
      v4 = *(_QWORD **)v24;
    }
    while ( *(_WORD *)(*(_QWORD *)v24 + 24LL) == 8 );
  }
  *a4 = v4;
  return i;
}
