// Function: sub_2E80380
// Address: 0x2e80380
//
void __fastcall sub_2E80380(__int64 a1, __int64 a2, void (*a3)(), __int64 a4)
{
  __int64 v4; // r14
  _QWORD *v6; // rbx
  __int64 v7; // rax
  _QWORD *v8; // r12
  __int64 v9; // rsi
  unsigned __int64 *v10; // rcx
  unsigned __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdi
  void (***v15)(void); // rdi
  void (*v16)(void); // rax
  _QWORD *v17; // rbx
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  __int64 v21; // r14
  unsigned __int64 *v22; // rbx
  unsigned __int64 *v23; // r12
  __int64 v24; // rbx
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi
  __int64 v27; // r14
  unsigned __int64 v28; // r12
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rdi
  __int64 v31; // r12
  __int64 v32; // rax
  __int64 *v33; // rbx
  __int64 *v34; // r13
  __int64 v35; // rax
  unsigned __int64 v36; // rdi

  v4 = a1 + 320;
  v6 = *(_QWORD **)(a1 + 328);
  *(_QWORD *)(a1 + 344) = 0;
  if ( v6 != (_QWORD *)(a1 + 320) )
  {
    do
    {
      v7 = v6[6];
      v8 = v6;
      v6[7] = v6 + 6;
      v9 = (__int64)v6;
      v6[6] = (unsigned __int64)(v6 + 6) | v7 & 7;
      v6 = (_QWORD *)v6[1];
      sub_2E31020(v4, v9);
      v10 = (unsigned __int64 *)v8[1];
      a2 = (__int64)v8;
      v11 = *v8 & 0xFFFFFFFFFFFFFFF8LL;
      *v10 = v11 | *v10 & 7;
      *(_QWORD *)(v11 + 8) = v10;
      *v8 &= 7uLL;
      v8[1] = 0;
      sub_2E79D60(v4, v8);
    }
    while ( (_QWORD *)v4 != v6 );
  }
  v12 = *(_QWORD *)(a1 + 96);
  if ( v12 != *(_QWORD *)(a1 + 104) )
    *(_QWORD *)(a1 + 104) = v12;
  *(_QWORD *)(a1 + 224) = 0;
  v13 = *(_QWORD *)(a1 + 552);
  *(_DWORD *)(a1 + 240) = 0;
  *(_QWORD *)(a1 + 312) = 0;
  if ( v13 != *(_QWORD *)(a1 + 560) )
    *(_QWORD *)(a1 + 560) = v13;
  *(_DWORD *)(a1 + 760) = 0;
  v14 = *(_QWORD *)(a1 + 32);
  if ( v14 )
    sub_2E78DE0(v14);
  v15 = *(void (****)(void))(a1 + 40);
  if ( v15 )
  {
    a3 = nullsub_1602;
    v16 = **v15;
    if ( v16 != nullsub_1602 )
      v16();
  }
  v17 = *(_QWORD **)(a1 + 48);
  v18 = v17[16];
  if ( (_QWORD *)v18 != v17 + 18 )
    _libc_free(v18);
  v19 = v17[12];
  if ( v19 )
  {
    a2 = v17[14] - v19;
    j_j___libc_free_0(v19);
  }
  v20 = v17[1];
  if ( v20 )
  {
    a2 = v17[3] - v20;
    j_j___libc_free_0(v20);
  }
  sub_2E7FFC0(*(_QWORD *)(a1 + 56), a2, (__int64)a3, a4);
  v21 = *(_QWORD *)(a1 + 64);
  if ( v21 )
  {
    v22 = *(unsigned __int64 **)(v21 + 16);
    v23 = *(unsigned __int64 **)(v21 + 8);
    if ( v22 != v23 )
    {
      do
      {
        if ( *v23 )
          j_j___libc_free_0(*v23);
        v23 += 4;
      }
      while ( v22 != v23 );
      v23 = *(unsigned __int64 **)(v21 + 8);
    }
    if ( v23 )
      j_j___libc_free_0((unsigned __int64)v23);
  }
  v24 = *(_QWORD *)(a1 + 88);
  if ( v24 )
  {
    v25 = *(_QWORD *)(v24 + 624);
    if ( v25 != v24 + 640 )
      _libc_free(v25);
    v26 = *(_QWORD *)(v24 + 512);
    if ( v26 != v24 + 528 )
      _libc_free(v26);
    v27 = *(_QWORD *)(v24 + 240);
    v28 = v27 + ((unsigned __int64)*(unsigned int *)(v24 + 248) << 6);
    if ( v27 != v28 )
    {
      do
      {
        v28 -= 64LL;
        v29 = *(_QWORD *)(v28 + 16);
        if ( v29 != v28 + 32 )
          _libc_free(v29);
      }
      while ( v27 != v28 );
      v28 = *(_QWORD *)(v24 + 240);
    }
    if ( v28 != v24 + 256 )
      _libc_free(v28);
    v30 = *(_QWORD *)(v24 + 160);
    if ( v30 != v24 + 176 )
      _libc_free(v30);
    sub_C7D6A0(*(_QWORD *)(v24 + 136), 16LL * *(unsigned int *)(v24 + 152), 8);
    sub_C7D6A0(*(_QWORD *)(v24 + 104), 24LL * *(unsigned int *)(v24 + 120), 8);
    sub_C7D6A0(*(_QWORD *)(v24 + 72), 16LL * *(unsigned int *)(v24 + 88), 8);
    sub_C7D6A0(*(_QWORD *)(v24 + 40), 16LL * *(unsigned int *)(v24 + 56), 8);
    sub_C7D6A0(*(_QWORD *)(v24 + 8), 16LL * *(unsigned int *)(v24 + 24), 8);
  }
  v31 = *(_QWORD *)(a1 + 80);
  if ( v31 )
  {
    v32 = *(unsigned int *)(v31 + 56);
    if ( (_DWORD)v32 )
    {
      v33 = *(__int64 **)(v31 + 40);
      v34 = &v33[9 * v32];
      do
      {
        while ( 1 )
        {
          v35 = *v33;
          BYTE1(v35) = BYTE1(*v33) & 0xEF;
          if ( v35 != -8192 && !*((_BYTE *)v33 + 36) )
            break;
          v33 += 9;
          if ( v34 == v33 )
            goto LABEL_48;
        }
        v36 = v33[2];
        v33 += 9;
        _libc_free(v36);
      }
      while ( v34 != v33 );
LABEL_48:
      v32 = *(unsigned int *)(v31 + 56);
    }
    sub_C7D6A0(*(_QWORD *)(v31 + 40), 72 * v32, 8);
    sub_C7D6A0(*(_QWORD *)(v31 + 8), 16LL * *(unsigned int *)(v31 + 24), 8);
  }
}
