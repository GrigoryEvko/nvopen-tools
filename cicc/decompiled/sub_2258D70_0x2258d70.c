// Function: sub_2258D70
// Address: 0x2258d70
//
void __fastcall sub_2258D70(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // r13
  unsigned __int64 v4; // r8
  __int64 v5; // r13
  __int64 v6; // r12
  _QWORD *v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi

  *(_QWORD *)a1 = &unk_4A08338;
  v2 = *(_QWORD *)(a1 + 280);
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  sub_2C835D0(a1 + 264);
  if ( *(_DWORD *)(a1 + 228) )
  {
    v3 = *(unsigned int *)(a1 + 224);
    v4 = *(_QWORD *)(a1 + 216);
    if ( (_DWORD)v3 )
    {
      v5 = 8 * v3;
      v6 = 0;
      do
      {
        v7 = *(_QWORD **)(v4 + v6);
        if ( v7 != (_QWORD *)-8LL && v7 )
        {
          sub_C7D6A0((__int64)v7, *v7 + 17LL, 8);
          v4 = *(_QWORD *)(a1 + 216);
        }
        v6 += 8;
      }
      while ( v5 != v6 );
    }
  }
  else
  {
    v4 = *(_QWORD *)(a1 + 216);
  }
  _libc_free(v4);
  v8 = *(_QWORD *)(a1 + 184);
  if ( v8 != a1 + 200 )
    j_j___libc_free_0(v8);
  v9 = *(_QWORD *)(a1 + 152);
  if ( v9 != a1 + 168 )
    j_j___libc_free_0(v9);
  v10 = *(_QWORD *)(a1 + 120);
  if ( v10 != a1 + 136 )
    j_j___libc_free_0(v10);
  v11 = *(_QWORD *)(a1 + 88);
  if ( v11 != a1 + 104 )
    j_j___libc_free_0(v11);
  v12 = *(_QWORD *)(a1 + 56);
  if ( v12 != a1 + 72 )
    j_j___libc_free_0(v12);
  v13 = *(_QWORD *)(a1 + 24);
  if ( v13 != a1 + 40 )
    j_j___libc_free_0(v13);
}
