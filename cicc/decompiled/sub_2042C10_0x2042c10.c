// Function: sub_2042C10
// Address: 0x2042c10
//
void __fastcall sub_2042C10(__int64 a1, __int64 a2)
{
  __int64 (*v4)(void); // rdx
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 (*v7)(); // rdx
  __int64 v8; // rax
  __int64 (*v9)(); // rax
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 (*v12)(); // rdx
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // rcx
  unsigned __int64 v16; // r12
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rcx
  unsigned __int64 v20; // rax
  _BYTE *v21; // rdx
  _BYTE *v22; // rdi
  _BYTE *v23; // rdx
  _BYTE *v24; // rdi
  _QWORD *v25; // rdi
  __int64 *v26; // r15
  __int64 *v27; // r12
  __int64 v28; // rsi
  __int64 (*v29)(); // rcx
  _DWORD *v30; // r14
  __int64 v31; // rax

  *(_BYTE *)(a1 + 12) = 0;
  *(_DWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)a1 = &unk_49FFF60;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = a1;
  v4 = *(__int64 (**)(void))(**(_QWORD **)(*(_QWORD *)(a2 + 256) + 16LL) + 128LL);
  v5 = 0;
  if ( v4 != sub_1D0B140 )
    v5 = v4();
  *(_QWORD *)(a1 + 152) = v5;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  v6 = *(_QWORD *)(*(_QWORD *)(a2 + 256) + 16LL);
  v7 = *(__int64 (**)())(*(_QWORD *)v6 + 112LL);
  v8 = 0;
  if ( v7 != sub_1D00B10 )
    v8 = ((__int64 (__fastcall *)(_QWORD))v7)(*(_QWORD *)(*(_QWORD *)(a2 + 256) + 16LL));
  *(_QWORD *)(a1 + 128) = v8;
  *(_QWORD *)(a1 + 136) = *(_QWORD *)(a2 + 320);
  v9 = *(__int64 (**)())(*(_QWORD *)v6 + 40LL);
  if ( v9 == sub_1D00B00 )
  {
    *(_QWORD *)(a1 + 144) = 0;
    BUG();
  }
  v10 = ((__int64 (__fastcall *)(__int64))v9)(v6);
  *(_QWORD *)(a1 + 144) = v10;
  v11 = v10;
  v12 = *(__int64 (**)())(*(_QWORD *)v10 + 944LL);
  v13 = 0;
  if ( v12 != sub_1E40480 )
    v13 = ((__int64 (__fastcall *)(__int64, __int64))v12)(v11, v6);
  v14 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 160) = v13;
  if ( v14 )
  {
    j___libc_free_0(*(_QWORD *)(v14 + 40));
    j_j___libc_free_0(v14, 64);
  }
  v15 = *(_QWORD *)(a1 + 96);
  v16 = (unsigned int)((__int64)(*(_QWORD *)(*(_QWORD *)(a1 + 128) + 264LL) - *(_QWORD *)(*(_QWORD *)(a1 + 128) + 256LL)) >> 3);
  v17 = (*(_QWORD *)(a1 + 104) - v15) >> 2;
  if ( v16 > v17 )
  {
    sub_C17A60(a1 + 96, v16 - v17);
  }
  else if ( v16 < v17 )
  {
    v18 = v15 + 4 * v16;
    if ( *(_QWORD *)(a1 + 104) != v18 )
      *(_QWORD *)(a1 + 104) = v18;
  }
  v19 = *(_QWORD *)(a1 + 72);
  v20 = (*(_QWORD *)(a1 + 80) - v19) >> 2;
  if ( v16 > v20 )
  {
    sub_C17A60(a1 + 72, v16 - v20);
  }
  else if ( v16 < v20 )
  {
    v31 = v19 + 4 * v16;
    if ( *(_QWORD *)(a1 + 80) != v31 )
      *(_QWORD *)(a1 + 80) = v31;
  }
  v21 = *(_BYTE **)(a1 + 104);
  v22 = *(_BYTE **)(a1 + 96);
  if ( v22 != v21 )
    memset(v22, 0, v21 - v22);
  v23 = *(_BYTE **)(a1 + 80);
  v24 = *(_BYTE **)(a1 + 72);
  if ( v24 != v23 )
    memset(v24, 0, v23 - v24);
  v25 = *(_QWORD **)(a1 + 128);
  v26 = (__int64 *)v25[33];
  v27 = (__int64 *)v25[32];
  if ( v27 != v26 )
  {
    while ( 1 )
    {
      v28 = *v27;
      v29 = *(__int64 (**)())(*v25 + 168LL);
      v30 = (_DWORD *)(*(_QWORD *)(a1 + 96) + 4LL * *(unsigned __int16 *)(*(_QWORD *)*v27 + 24LL));
      if ( v29 == sub_1D00B20 )
      {
        ++v27;
        *v30 = 0;
        if ( v26 == v27 )
          break;
      }
      else
      {
        ++v27;
        *v30 = ((__int64 (__fastcall *)(_QWORD *, __int64, _QWORD, __int64 (*)(), __int64 (*)()))v29)(
                 v25,
                 v28,
                 *(_QWORD *)(a2 + 256),
                 v29,
                 sub_1D00B20);
        if ( v26 == v27 )
          break;
      }
      v25 = *(_QWORD **)(a1 + 128);
    }
  }
  *(_QWORD *)(a1 + 192) = 0;
}
