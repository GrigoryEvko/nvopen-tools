// Function: sub_2E246F0
// Address: 0x2e246f0
//
__int64 __fastcall sub_2E246F0(__int64 a1)
{
  __int64 v2; // rsi
  unsigned __int64 *v3; // rbx
  unsigned __int64 *v4; // r12
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  _QWORD *v8; // rbx
  _QWORD *v9; // r14
  unsigned __int64 v10; // rdi
  _QWORD *v11; // r15
  _QWORD *v12; // r12
  unsigned __int64 v13; // rdi
  _QWORD *v14; // rbx
  unsigned __int64 v15; // rdi

  v2 = 16LL * *(unsigned int *)(a1 + 400);
  *(_QWORD *)a1 = &unk_4A28670;
  sub_C7D6A0(*(_QWORD *)(a1 + 384), v2, 8);
  v3 = *(unsigned __int64 **)(a1 + 360);
  v4 = *(unsigned __int64 **)(a1 + 352);
  if ( v3 != v4 )
  {
    do
    {
      if ( (unsigned __int64 *)*v4 != v4 + 2 )
        _libc_free(*v4);
      v4 += 4;
    }
    while ( v3 != v4 );
    v4 = *(unsigned __int64 **)(a1 + 352);
  }
  if ( v4 )
    j_j___libc_free_0((unsigned __int64)v4);
  v5 = *(_QWORD *)(a1 + 328);
  if ( v5 )
    j_j___libc_free_0(v5);
  v6 = *(_QWORD *)(a1 + 304);
  if ( v6 )
    j_j___libc_free_0(v6);
  v7 = *(_QWORD *)(a1 + 248);
  if ( v7 )
    j_j___libc_free_0(v7);
  v8 = *(_QWORD **)(a1 + 216);
  v9 = (_QWORD *)(a1 + 216);
  while ( v8 != v9 )
  {
    v10 = (unsigned __int64)v8;
    v8 = (_QWORD *)*v8;
    j_j___libc_free_0(v10);
  }
  v11 = *(_QWORD **)(a1 + 200);
  v12 = &v11[7 * *(unsigned int *)(a1 + 208)];
  if ( v11 != v12 )
  {
    do
    {
      v13 = *(v12 - 3);
      v12 -= 7;
      if ( v13 )
        j_j___libc_free_0(v13);
      v14 = (_QWORD *)*v12;
      while ( v12 != v14 )
      {
        v15 = (unsigned __int64)v14;
        v14 = (_QWORD *)*v14;
        j_j___libc_free_0(v15);
      }
    }
    while ( v11 != v12 );
    v12 = *(_QWORD **)(a1 + 200);
  }
  if ( v9 != v12 )
    _libc_free((unsigned __int64)v12);
  *(_QWORD *)a1 = &unk_49DAF80;
  return sub_BB9100(a1);
}
