// Function: sub_B83890
// Address: 0xb83890
//
__int64 __fastcall sub_B83890(__int64 a1)
{
  _QWORD *v1; // r12
  _QWORD *i; // r13
  _QWORD *v4; // r12
  _QWORD *j; // r13
  __int64 v6; // rsi
  __int64 *v7; // r14
  __int64 *v8; // r12
  __int64 k; // rax
  __int64 v10; // rdi
  unsigned int v11; // ecx
  __int64 *v12; // r12
  __int64 *v13; // r13
  __int64 v14; // rdi
  __int64 v15; // rdi
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v19; // r13
  __int64 v20; // rdi
  __int64 v21; // rsi
  __int64 v22; // rdi
  __int64 v23; // rdi
  __int64 result; // rax
  __int64 v25; // rdi

  v1 = *(_QWORD **)(a1 + 32);
  *(_QWORD *)a1 = &unk_49DA9C0;
  for ( i = &v1[*(unsigned int *)(a1 + 40)]; i != v1; ++v1 )
  {
    if ( *v1 )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v1 + 8LL))(*v1);
  }
  v4 = *(_QWORD **)(a1 + 256);
  for ( j = &v4[*(unsigned int *)(a1 + 264)]; j != v4; ++v4 )
  {
    if ( *v4 )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v4 + 8LL))(*v4);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 696), 16LL * *(unsigned int *)(a1 + 712), 8);
  v6 = 16LL * *(unsigned int *)(a1 + 680);
  sub_C7D6A0(*(_QWORD *)(a1 + 664), v6, 8);
  sub_B83600(a1 + 560);
  v7 = *(__int64 **)(a1 + 576);
  v8 = &v7[*(unsigned int *)(a1 + 584)];
  if ( v7 != v8 )
  {
    for ( k = *(_QWORD *)(a1 + 576); ; k = *(_QWORD *)(a1 + 576) )
    {
      v10 = *v7;
      v11 = (unsigned int)(((__int64)v7 - k) >> 3) >> 7;
      v6 = 4096LL << v11;
      if ( v11 >= 0x1E )
        v6 = 0x40000000000LL;
      ++v7;
      sub_C7D6A0(v10, v6, 16);
      if ( v8 == v7 )
        break;
    }
  }
  v12 = *(__int64 **)(a1 + 624);
  v13 = &v12[2 * *(unsigned int *)(a1 + 632)];
  if ( v12 != v13 )
  {
    do
    {
      v6 = v12[1];
      v14 = *v12;
      v12 += 2;
      sub_C7D6A0(v14, v6, 16);
    }
    while ( v13 != v12 );
    v13 = *(__int64 **)(a1 + 624);
  }
  if ( v13 != (__int64 *)(a1 + 640) )
    _libc_free(v13, v6);
  v15 = *(_QWORD *)(a1 + 576);
  if ( v15 != a1 + 592 )
    _libc_free(v15, v6);
  sub_C65770(a1 + 544);
  if ( (*(_BYTE *)(a1 + 408) & 1) == 0 )
  {
    v6 = 16LL * *(unsigned int *)(a1 + 424);
    sub_C7D6A0(*(_QWORD *)(a1 + 416), v6, 8);
  }
  v16 = *(_QWORD *)(a1 + 256);
  if ( v16 != a1 + 272 )
    _libc_free(v16, v6);
  v17 = *(unsigned int *)(a1 + 248);
  if ( (_DWORD)v17 )
  {
    v18 = *(_QWORD *)(a1 + 232);
    v19 = v18 + 104 * v17;
    do
    {
      while ( *(_QWORD *)v18 == -8192 || *(_QWORD *)v18 == -4096 || *(_BYTE *)(v18 + 36) )
      {
        v18 += 104;
        if ( v19 == v18 )
          goto LABEL_33;
      }
      v20 = *(_QWORD *)(v18 + 16);
      v18 += 104;
      _libc_free(v20, v6);
    }
    while ( v19 != v18 );
LABEL_33:
    v17 = *(unsigned int *)(a1 + 248);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 232), 104 * v17, 8);
  v21 = 16LL * *(unsigned int *)(a1 + 216);
  sub_C7D6A0(*(_QWORD *)(a1 + 200), v21, 8);
  v22 = *(_QWORD *)(a1 + 112);
  if ( v22 != a1 + 128 )
    _libc_free(v22, v21);
  v23 = *(_QWORD *)(a1 + 32);
  result = a1 + 48;
  if ( v23 != a1 + 48 )
    result = _libc_free(v23, v21);
  v25 = *(_QWORD *)(a1 + 8);
  if ( v25 )
    return j_j___libc_free_0(v25, *(_QWORD *)(a1 + 24) - v25);
  return result;
}
