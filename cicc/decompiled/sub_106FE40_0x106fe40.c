// Function: sub_106FE40
// Address: 0x106fe40
//
__int64 __fastcall sub_106FE40(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r15
  _QWORD *v4; // r13
  _QWORD *v5; // rbx
  _QWORD *v6; // r12
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // rdi
  __int64 v15; // rdi
  __int64 v16; // rax
  _QWORD *v17; // rbx
  _QWORD *v18; // r12
  __int64 v19; // rdi
  __int64 v20; // rsi
  __int64 v21; // rdi

  v3 = *(_QWORD **)(a1 + 2032);
  v4 = *(_QWORD **)(a1 + 2024);
  *(_QWORD *)a1 = &unk_49E60C0;
  if ( v3 != v4 )
  {
    do
    {
      v5 = (_QWORD *)v4[1];
      v6 = (_QWORD *)*v4;
      if ( v5 != (_QWORD *)*v4 )
      {
        do
        {
          if ( (_QWORD *)*v6 != v6 + 2 )
          {
            a2 = v6[2] + 1LL;
            j_j___libc_free_0(*v6, a2);
          }
          v6 += 4;
        }
        while ( v5 != v6 );
        v6 = (_QWORD *)*v4;
      }
      if ( v6 )
      {
        a2 = v4[2] - (_QWORD)v6;
        j_j___libc_free_0(v6, a2);
      }
      v4 += 3;
    }
    while ( v3 != v4 );
    v4 = *(_QWORD **)(a1 + 2024);
  }
  if ( v4 )
  {
    a2 = *(_QWORD *)(a1 + 2040) - (_QWORD)v4;
    j_j___libc_free_0(v4, a2);
  }
  v7 = *(_QWORD *)(a1 + 400);
  v8 = v7 + 48LL * *(unsigned int *)(a1 + 408);
  if ( v7 != v8 )
  {
    do
    {
      v8 -= 48;
      v9 = *(_QWORD *)(v8 + 8);
      if ( v9 != v8 + 24 )
        _libc_free(v9, a2);
    }
    while ( v7 != v8 );
    v8 = *(_QWORD *)(a1 + 400);
  }
  if ( v8 != a1 + 416 )
    _libc_free(v8, a2);
  v10 = *(_QWORD *)(a1 + 368);
  if ( v10 )
  {
    a2 = *(_QWORD *)(a1 + 384) - v10;
    j_j___libc_free_0(v10, a2);
  }
  v11 = *(_QWORD *)(a1 + 344);
  if ( v11 )
  {
    a2 = *(_QWORD *)(a1 + 360) - v11;
    j_j___libc_free_0(v11, a2);
  }
  v12 = *(_QWORD *)(a1 + 320);
  if ( v12 )
  {
    a2 = *(_QWORD *)(a1 + 336) - v12;
    j_j___libc_free_0(v12, a2);
  }
  sub_C0BF30(a1 + 272);
  v13 = *(_QWORD *)(a1 + 256);
  if ( a1 + 272 != v13 )
    _libc_free(v13, a2);
  sub_C7D6A0(*(_QWORD *)(a1 + 232), 16LL * *(unsigned int *)(a1 + 248), 8);
  v14 = *(_QWORD *)(a1 + 200);
  if ( v14 )
    j_j___libc_free_0(v14, *(_QWORD *)(a1 + 216) - v14);
  sub_C7D6A0(*(_QWORD *)(a1 + 176), 16LL * *(unsigned int *)(a1 + 192), 8);
  v15 = *(_QWORD *)(a1 + 144);
  if ( v15 )
    j_j___libc_free_0(v15, *(_QWORD *)(a1 + 160) - v15);
  v16 = *(unsigned int *)(a1 + 136);
  if ( (_DWORD)v16 )
  {
    v17 = *(_QWORD **)(a1 + 120);
    v18 = &v17[4 * v16];
    do
    {
      if ( *v17 != -8192 && *v17 != -4096 )
      {
        v19 = v17[1];
        if ( v19 )
          j_j___libc_free_0(v19, v17[3] - v19);
      }
      v17 += 4;
    }
    while ( v18 != v17 );
    LODWORD(v16) = *(_DWORD *)(a1 + 136);
  }
  v20 = 32LL * (unsigned int)v16;
  sub_C7D6A0(*(_QWORD *)(a1 + 120), v20, 8);
  v21 = *(_QWORD *)(a1 + 104);
  if ( v21 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v21 + 8LL))(v21);
  return sub_E8EC10(a1, v20);
}
