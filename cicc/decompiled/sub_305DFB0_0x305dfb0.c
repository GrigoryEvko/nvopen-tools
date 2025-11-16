// Function: sub_305DFB0
// Address: 0x305dfb0
//
void __fastcall sub_305DFB0(unsigned __int64 a1)
{
  unsigned __int64 v2; // r8
  __int64 v3; // r13
  __int64 v4; // r13
  __int64 v5; // rbx
  _QWORD *v6; // rdi

  v2 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)a1 = &unk_4A30FA0;
  if ( *(_DWORD *)(a1 + 20) )
  {
    v3 = *(unsigned int *)(a1 + 16);
    if ( (_DWORD)v3 )
    {
      v4 = 8 * v3;
      v5 = 0;
      do
      {
        v6 = *(_QWORD **)(v2 + v5);
        if ( v6 != (_QWORD *)-8LL && v6 )
        {
          sub_C7D6A0((__int64)v6, *v6 + 17LL, 8);
          v2 = *(_QWORD *)(a1 + 8);
        }
        v5 += 8;
      }
      while ( v4 != v5 );
    }
  }
  _libc_free(v2);
  j_j___libc_free_0(a1);
}
