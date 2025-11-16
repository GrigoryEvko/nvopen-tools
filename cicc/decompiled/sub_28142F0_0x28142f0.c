// Function: sub_28142F0
// Address: 0x28142f0
//
_QWORD *__fastcall sub_28142F0(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  int v9; // esi
  _QWORD *v10; // r13
  __int64 v11; // rcx
  char v12; // al
  __int64 v13; // rdi
  __int64 v14; // rax
  int *v15; // r12
  __int64 v16; // rbx
  __int64 v17; // r14
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rax
  __int64 v22; // rdx
  int v23; // eax
  __int64 v24; // rdi

  v4 = (_QWORD *)sub_22077B0(0x1A8u);
  v9 = *(_DWORD *)(a1 + 88);
  v10 = v4;
  v4[4] = *(_QWORD *)(a1 + 32);
  v4[5] = *(_QWORD *)(a1 + 40);
  v4[6] = *(_QWORD *)(a1 + 48);
  v4[7] = *(_QWORD *)(a1 + 56);
  v4[8] = *(_QWORD *)(a1 + 64);
  v4[9] = *(_QWORD *)(a1 + 72);
  v4[10] = v4 + 12;
  v4[11] = 0x1000000000LL;
  if ( v9 )
    sub_2813E60((__int64)(v4 + 10), a1 + 80, v5, v6, v7, v8);
  v11 = *(unsigned int *)(a1 + 232);
  v10[28] = v10 + 30;
  v10[29] = 0x1000000000LL;
  if ( (_DWORD)v11 )
    sub_2813E60((__int64)(v10 + 28), a1 + 224, v5, v11, v7, v8);
  v12 = *(_BYTE *)(a1 + 368);
  v13 = *(_QWORD *)(a1 + 24);
  v10[1] = a2;
  v10[2] = 0;
  *((_BYTE *)v10 + 368) = v12;
  v14 = *(_QWORD *)(a1 + 376);
  v10[3] = 0;
  v10[47] = v14;
  v10[48] = *(_QWORD *)(a1 + 384);
  *((_WORD *)v10 + 196) = *(_WORD *)(a1 + 392);
  v10[50] = *(_QWORD *)(a1 + 400);
  v10[51] = *(_QWORD *)(a1 + 408);
  v10[52] = *(_QWORD *)(a1 + 416);
  *(_DWORD *)v10 = *(_DWORD *)a1;
  if ( v13 )
    v10[3] = sub_28142F0(v13, v10);
  v15 = *(int **)(a1 + 16);
  if ( v15 )
  {
    v16 = (__int64)v10;
    do
    {
      v17 = v16;
      v16 = sub_22077B0(0x1A8u);
      *(_QWORD *)(v16 + 32) = *((_QWORD *)v15 + 4);
      *(_QWORD *)(v16 + 40) = *((_QWORD *)v15 + 5);
      *(_QWORD *)(v16 + 48) = *((_QWORD *)v15 + 6);
      *(_QWORD *)(v16 + 56) = *((_QWORD *)v15 + 7);
      *(_QWORD *)(v16 + 64) = *((_QWORD *)v15 + 8);
      v21 = *((_QWORD *)v15 + 9);
      *(_QWORD *)(v16 + 88) = 0x1000000000LL;
      *(_QWORD *)(v16 + 72) = v21;
      *(_QWORD *)(v16 + 80) = v16 + 96;
      v22 = (unsigned int)v15[22];
      if ( (_DWORD)v22 )
        sub_2813E60(v16 + 80, (__int64)(v15 + 20), v22, v18, v19, v20);
      *(_QWORD *)(v16 + 232) = 0x1000000000LL;
      *(_QWORD *)(v16 + 224) = v16 + 240;
      if ( v15[58] )
        sub_2813E60(v16 + 224, (__int64)(v15 + 56), v22, v18, v19, v20);
      *(_BYTE *)(v16 + 368) = *((_BYTE *)v15 + 368);
      *(_QWORD *)(v16 + 376) = *((_QWORD *)v15 + 47);
      *(_QWORD *)(v16 + 384) = *((_QWORD *)v15 + 48);
      *(_WORD *)(v16 + 392) = *((_WORD *)v15 + 196);
      *(_QWORD *)(v16 + 400) = *((_QWORD *)v15 + 50);
      *(_QWORD *)(v16 + 408) = *((_QWORD *)v15 + 51);
      *(_QWORD *)(v16 + 416) = *((_QWORD *)v15 + 52);
      v23 = *v15;
      *(_QWORD *)(v16 + 16) = 0;
      *(_DWORD *)v16 = v23;
      *(_QWORD *)(v16 + 24) = 0;
      *(_QWORD *)(v17 + 16) = v16;
      *(_QWORD *)(v16 + 8) = v17;
      v24 = *((_QWORD *)v15 + 3);
      if ( v24 )
        *(_QWORD *)(v16 + 24) = sub_28142F0(v24, v16);
      v15 = (int *)*((_QWORD *)v15 + 2);
    }
    while ( v15 );
  }
  return v10;
}
