// Function: sub_19C07E0
// Address: 0x19c07e0
//
__int64 __fastcall sub_19C07E0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  __int64 v4; // r13
  _QWORD *v5; // rbx
  _QWORD *v6; // r13
  __int64 v7; // rax
  unsigned __int64 *v8; // rax
  unsigned __int64 *v9; // r14
  __int64 v10; // rdi

  *(_QWORD *)a1 = off_49F46C0;
  v2 = *(_QWORD *)(a1 + 400);
  if ( v2 )
    j_j___libc_free_0(v2, *(_QWORD *)(a1 + 416) - v2);
  v3 = *(_QWORD *)(a1 + 376);
  if ( v3 )
    j_j___libc_free_0(v3, *(_QWORD *)(a1 + 392) - v3);
  v4 = *(unsigned int *)(a1 + 368);
  if ( (_DWORD)v4 )
  {
    v5 = *(_QWORD **)(a1 + 352);
    v6 = &v5[2 * v4];
    do
    {
      if ( *v5 != -16 && *v5 != -8 )
      {
        v7 = v5[1];
        if ( (v7 & 4) != 0 )
        {
          v8 = (unsigned __int64 *)(v7 & 0xFFFFFFFFFFFFFFF8LL);
          v9 = v8;
          if ( v8 )
          {
            if ( (unsigned __int64 *)*v8 != v8 + 2 )
              _libc_free(*v8);
            j_j___libc_free_0(v9, 48);
          }
        }
      }
      v5 += 2;
    }
    while ( v6 != v5 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 352));
  sub_19C0500(*(_QWORD *)(a1 + 232));
  v10 = *(_QWORD *)(a1 + 192);
  if ( v10 )
    j_j___libc_free_0(v10, *(_QWORD *)(a1 + 208) - v10);
  *(_QWORD *)a1 = &unk_49EAEF0;
  sub_16366C0((_QWORD *)a1);
  return j_j___libc_free_0(a1, 432);
}
