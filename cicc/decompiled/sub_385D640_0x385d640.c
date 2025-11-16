// Function: sub_385D640
// Address: 0x385d640
//
void __fastcall sub_385D640(__int64 a1)
{
  unsigned __int64 *v2; // rbx
  unsigned __int64 *v3; // r14
  unsigned __int64 v4; // rdx
  _QWORD *v5; // r15
  _QWORD *v6; // r12
  __int64 v7; // rax
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  _QWORD *v11; // rbx
  _QWORD *i; // r14
  __int64 v13; // rax
  void *v14; // [rsp+10h] [rbp-90h] BYREF
  __int64 v15; // [rsp+18h] [rbp-88h] BYREF
  __int64 v16; // [rsp+28h] [rbp-78h]
  void *v17; // [rsp+40h] [rbp-60h] BYREF
  __int64 v18; // [rsp+48h] [rbp-58h] BYREF
  __int64 v19; // [rsp+58h] [rbp-48h]

  sub_1359CD0(a1 + 320);
  if ( *(_DWORD *)(a1 + 368) )
  {
    sub_1359800(&v14, -8, 0);
    sub_1359800(&v17, -16, 0);
    v11 = *(_QWORD **)(a1 + 352);
    for ( i = &v11[6 * *(unsigned int *)(a1 + 368)]; i != v11; v11 += 6 )
    {
      v13 = v11[3];
      *v11 = &unk_49EE2B0;
      if ( v13 != 0 && v13 != -8 && v13 != -16 )
        sub_1649B30(v11 + 1);
    }
    v17 = &unk_49EE2B0;
    if ( v19 != 0 && v19 != -8 && v19 != -16 )
      sub_1649B30(&v18);
    v14 = &unk_49EE2B0;
    if ( v16 != 0 && v16 != -8 && v16 != -16 )
      sub_1649B30(&v15);
  }
  j___libc_free_0(*(_QWORD *)(a1 + 352));
  v2 = *(unsigned __int64 **)(a1 + 336);
  while ( (unsigned __int64 *)(a1 + 328) != v2 )
  {
    v3 = v2;
    v2 = (unsigned __int64 *)v2[1];
    v4 = *v3 & 0xFFFFFFFFFFFFFFF8LL;
    *v2 = v4 | *v2 & 7;
    *(_QWORD *)(v4 + 8) = v2;
    v5 = (_QWORD *)v3[6];
    v6 = (_QWORD *)v3[5];
    *v3 &= 7u;
    v3[1] = 0;
    if ( v5 != v6 )
    {
      do
      {
        v7 = v6[2];
        if ( v7 != -8 && v7 != 0 && v7 != -16 )
          sub_1649B30(v6);
        v6 += 3;
      }
      while ( v5 != v6 );
      v6 = (_QWORD *)v3[5];
    }
    if ( v6 )
      j_j___libc_free_0((unsigned __int64)v6);
    j_j___libc_free_0((unsigned __int64)v3);
  }
  v8 = *(_QWORD *)(a1 + 168);
  if ( v8 != *(_QWORD *)(a1 + 160) )
    _libc_free(v8);
  v9 = *(_QWORD *)(a1 + 72);
  if ( v9 != a1 + 88 )
    _libc_free(v9);
  v10 = *(_QWORD *)(a1 + 32);
  if ( v10 )
    j_j___libc_free_0(v10);
  j___libc_free_0(*(_QWORD *)(a1 + 8));
}
