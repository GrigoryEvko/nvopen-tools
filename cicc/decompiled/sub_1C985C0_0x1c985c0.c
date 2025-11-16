// Function: sub_1C985C0
// Address: 0x1c985c0
//
__int64 __fastcall sub_1C985C0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rbx
  __int64 v4; // r13
  __int64 v5; // rdi
  __int64 v6; // r13
  __int64 v7; // r14
  __int64 v8; // rbx
  __int64 v9; // rdi
  __int64 v10; // rax
  _QWORD *v11; // rbx
  _QWORD *v12; // r13
  unsigned __int64 v13; // rdi
  _QWORD *v14; // rbx
  _QWORD *v15; // rdi
  __int64 v16; // rdi
  __int64 result; // rax
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 v20; // rdi
  __int64 v21; // rdi

  sub_1C96740(*(_QWORD *)(a1 + 512));
  sub_1C973F0(*(_QWORD **)(a1 + 464));
  v2 = *(_QWORD *)(a1 + 424);
  if ( v2 )
    j_j___libc_free_0(v2, *(_QWORD *)(a1 + 440) - v2);
  v3 = *(_QWORD *)(a1 + 392);
  while ( v3 )
  {
    v4 = v3;
    sub_1C96F50(*(_QWORD **)(v3 + 24));
    v5 = *(_QWORD *)(v3 + 40);
    v3 = *(_QWORD *)(v3 + 16);
    if ( v5 )
      j_j___libc_free_0(v5, *(_QWORD *)(v4 + 56) - v5);
    j_j___libc_free_0(v4, 64);
  }
  sub_1C96D30(*(_QWORD *)(a1 + 320));
  v6 = *(_QWORD *)(a1 + 272);
  while ( v6 )
  {
    v7 = v6;
    sub_1C97640(*(_QWORD **)(v6 + 24));
    v8 = *(_QWORD *)(v6 + 64);
    v6 = *(_QWORD *)(v6 + 16);
    while ( v8 )
    {
      sub_1C97470(*(_QWORD *)(v8 + 24));
      v9 = v8;
      v8 = *(_QWORD *)(v8 + 16);
      j_j___libc_free_0(v9, 40);
    }
    j_j___libc_free_0(v7, 96);
  }
  v10 = *(unsigned int *)(a1 + 248);
  if ( (_DWORD)v10 )
  {
    v11 = *(_QWORD **)(a1 + 232);
    v12 = &v11[6 * v10];
    do
    {
      if ( *v11 != -8 && *v11 != -16 )
      {
        v13 = v11[1];
        if ( (_QWORD *)v13 != v11 + 3 )
          _libc_free(v13);
      }
      v11 += 6;
    }
    while ( v12 != v11 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 232));
  sub_1C96000(*(_QWORD *)(a1 + 192));
  v14 = *(_QWORD **)(a1 + 136);
  while ( v14 )
  {
    v15 = v14;
    v14 = (_QWORD *)*v14;
    j_j___libc_free_0(v15, 32);
  }
  memset(*(void **)(a1 + 120), 0, 8LL * *(_QWORD *)(a1 + 128));
  v16 = *(_QWORD *)(a1 + 120);
  result = a1 + 168;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  if ( v16 != a1 + 168 )
    result = j_j___libc_free_0(v16, 8LL * *(_QWORD *)(a1 + 128));
  v18 = *(_QWORD *)(a1 + 96);
  if ( v18 )
    result = j_j___libc_free_0(v18, *(_QWORD *)(a1 + 112) - v18);
  v19 = *(_QWORD *)(a1 + 72);
  if ( v19 )
    result = j_j___libc_free_0(v19, *(_QWORD *)(a1 + 88) - v19);
  v20 = *(_QWORD *)(a1 + 48);
  if ( v20 )
    result = j_j___libc_free_0(v20, *(_QWORD *)(a1 + 64) - v20);
  v21 = *(_QWORD *)(a1 + 24);
  if ( v21 )
    return j_j___libc_free_0(v21, *(_QWORD *)(a1 + 40) - v21);
  return result;
}
