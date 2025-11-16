// Function: sub_142D3E0
// Address: 0x142d3e0
//
__int64 __fastcall sub_142D3E0(__int64 a1)
{
  unsigned __int64 *v2; // rbx
  unsigned __int64 *v3; // r13
  unsigned __int64 v4; // rdi
  unsigned __int64 *v5; // rbx
  unsigned __int64 v6; // r13
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  __int64 v9; // rbx
  __int64 v10; // r13
  __int64 v11; // rdi
  __int64 v12; // rbx
  __int64 v13; // r13
  __int64 v14; // rdi
  __int64 v15; // rbx
  __int64 v16; // rdi
  __int64 v17; // r13
  unsigned __int64 v18; // r8
  __int64 v19; // r13
  __int64 v20; // rbx
  unsigned __int64 v21; // rdi

  v2 = *(unsigned __int64 **)(a1 + 296);
  v3 = &v2[*(unsigned int *)(a1 + 304)];
  while ( v3 != v2 )
  {
    v4 = *v2++;
    _libc_free(v4);
  }
  v5 = *(unsigned __int64 **)(a1 + 344);
  v6 = (unsigned __int64)&v5[2 * *(unsigned int *)(a1 + 352)];
  if ( v5 != (unsigned __int64 *)v6 )
  {
    do
    {
      v7 = *v5;
      v5 += 2;
      _libc_free(v7);
    }
    while ( (unsigned __int64 *)v6 != v5 );
    v6 = *(_QWORD *)(a1 + 344);
  }
  if ( v6 != a1 + 360 )
    _libc_free(v6);
  v8 = *(_QWORD *)(a1 + 296);
  if ( v8 != a1 + 312 )
    _libc_free(v8);
  v9 = *(_QWORD *)(a1 + 248);
  while ( v9 )
  {
    v10 = v9;
    sub_142C2C0(*(_QWORD **)(v9 + 24));
    v11 = *(_QWORD *)(v9 + 32);
    v9 = *(_QWORD *)(v9 + 16);
    if ( v11 != v10 + 48 )
      j_j___libc_free_0(v11, *(_QWORD *)(v10 + 48) + 1LL);
    j_j___libc_free_0(v10, 64);
  }
  v12 = *(_QWORD *)(a1 + 200);
  while ( v12 )
  {
    v13 = v12;
    sub_142C2C0(*(_QWORD **)(v12 + 24));
    v14 = *(_QWORD *)(v12 + 32);
    v12 = *(_QWORD *)(v12 + 16);
    if ( v14 != v13 + 48 )
      j_j___libc_free_0(v14, *(_QWORD *)(v13 + 48) + 1LL);
    j_j___libc_free_0(v13, 64);
  }
  v15 = *(_QWORD *)(a1 + 144);
  while ( v15 )
  {
    sub_142BD00(*(_QWORD *)(v15 + 24));
    v16 = v15;
    v15 = *(_QWORD *)(v15 + 16);
    j_j___libc_free_0(v16, 48);
  }
  sub_142C250(*(_QWORD **)(a1 + 96));
  if ( *(_DWORD *)(a1 + 60) )
  {
    v17 = *(unsigned int *)(a1 + 56);
    v18 = *(_QWORD *)(a1 + 48);
    if ( (_DWORD)v17 )
    {
      v19 = 8 * v17;
      v20 = 0;
      do
      {
        v21 = *(_QWORD *)(v18 + v20);
        if ( v21 != -8 && v21 )
        {
          _libc_free(v21);
          v18 = *(_QWORD *)(a1 + 48);
        }
        v20 += 8;
      }
      while ( v20 != v19 );
    }
  }
  else
  {
    v18 = *(_QWORD *)(a1 + 48);
  }
  _libc_free(v18);
  return sub_142B9F0(*(_QWORD **)(a1 + 16));
}
