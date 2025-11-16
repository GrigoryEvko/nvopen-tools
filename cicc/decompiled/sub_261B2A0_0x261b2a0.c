// Function: sub_261B2A0
// Address: 0x261b2a0
//
__int64 __fastcall sub_261B2A0(int *a1, __int64 a2)
{
  __int64 v4; // rax
  _BYTE *v5; // rsi
  __int64 v6; // r14
  __int64 v7; // rdx
  __int64 v8; // rdi
  _QWORD *v9; // rax
  _QWORD *v10; // rcx
  _QWORD *v11; // rdx
  _QWORD *v12; // rax
  _QWORD *v13; // rdx
  __int64 v14; // rax
  int v15; // eax
  __int64 v16; // rdi
  int *v17; // r12
  __int64 v18; // rbx
  __int64 v19; // r13
  __int64 v20; // rdi
  _QWORD *v21; // rax
  _QWORD *v22; // rcx
  _QWORD *v23; // rdx
  _QWORD *v24; // rax
  _QWORD *v25; // rdx
  __int64 v26; // rax
  int v27; // eax
  __int64 v28; // rdi

  v4 = sub_22077B0(0x80u);
  v5 = (_BYTE *)*((_QWORD *)a1 + 6);
  v6 = v4;
  v7 = (__int64)&v5[*((_QWORD *)a1 + 7)];
  *(_QWORD *)(v4 + 32) = *((_QWORD *)a1 + 4);
  *(_DWORD *)(v4 + 40) = a1[10];
  *(_QWORD *)(v4 + 48) = v4 + 64;
  sub_261A960((__int64 *)(v4 + 48), v5, v7);
  v8 = *((_QWORD *)a1 + 12);
  *(_DWORD *)(v6 + 88) = 0;
  *(_QWORD *)(v6 + 96) = 0;
  *(_QWORD *)(v6 + 104) = v6 + 88;
  *(_QWORD *)(v6 + 112) = v6 + 88;
  *(_QWORD *)(v6 + 120) = 0;
  if ( v8 )
  {
    v9 = sub_261B080(v8, v6 + 88);
    v10 = v9;
    do
    {
      v11 = v9;
      v9 = (_QWORD *)v9[2];
    }
    while ( v9 );
    *(_QWORD *)(v6 + 104) = v11;
    v12 = v10;
    do
    {
      v13 = v12;
      v12 = (_QWORD *)v12[3];
    }
    while ( v12 );
    v14 = *((_QWORD *)a1 + 15);
    *(_QWORD *)(v6 + 112) = v13;
    *(_QWORD *)(v6 + 96) = v10;
    *(_QWORD *)(v6 + 120) = v14;
  }
  v15 = *a1;
  v16 = *((_QWORD *)a1 + 3);
  *(_QWORD *)(v6 + 8) = a2;
  *(_QWORD *)(v6 + 16) = 0;
  *(_DWORD *)v6 = v15;
  *(_QWORD *)(v6 + 24) = 0;
  if ( v16 )
    *(_QWORD *)(v6 + 24) = sub_261B2A0(v16, v6);
  v17 = (int *)*((_QWORD *)a1 + 2);
  if ( v17 )
  {
    v18 = v6;
    do
    {
      v19 = v18;
      v18 = sub_22077B0(0x80u);
      *(_QWORD *)(v18 + 32) = *((_QWORD *)v17 + 4);
      *(_DWORD *)(v18 + 40) = v17[10];
      *(_QWORD *)(v18 + 48) = v18 + 64;
      sub_261A960((__int64 *)(v18 + 48), *((_BYTE **)v17 + 6), *((_QWORD *)v17 + 6) + *((_QWORD *)v17 + 7));
      *(_DWORD *)(v18 + 88) = 0;
      *(_QWORD *)(v18 + 96) = 0;
      *(_QWORD *)(v18 + 104) = v18 + 88;
      *(_QWORD *)(v18 + 112) = v18 + 88;
      *(_QWORD *)(v18 + 120) = 0;
      v20 = *((_QWORD *)v17 + 12);
      if ( v20 )
      {
        v21 = sub_261B080(v20, v18 + 88);
        v22 = v21;
        do
        {
          v23 = v21;
          v21 = (_QWORD *)v21[2];
        }
        while ( v21 );
        *(_QWORD *)(v18 + 104) = v23;
        v24 = v22;
        do
        {
          v25 = v24;
          v24 = (_QWORD *)v24[3];
        }
        while ( v24 );
        *(_QWORD *)(v18 + 112) = v25;
        v26 = *((_QWORD *)v17 + 15);
        *(_QWORD *)(v18 + 96) = v22;
        *(_QWORD *)(v18 + 120) = v26;
      }
      v27 = *v17;
      *(_QWORD *)(v18 + 16) = 0;
      *(_QWORD *)(v18 + 24) = 0;
      *(_DWORD *)v18 = v27;
      *(_QWORD *)(v19 + 16) = v18;
      *(_QWORD *)(v18 + 8) = v19;
      v28 = *((_QWORD *)v17 + 3);
      if ( v28 )
        *(_QWORD *)(v18 + 24) = sub_261B2A0(v28, v18);
      v17 = (int *)*((_QWORD *)v17 + 2);
    }
    while ( v17 );
  }
  return v6;
}
