// Function: sub_230D0B0
// Address: 0x230d0b0
//
void __fastcall sub_230D0B0(unsigned __int64 a1)
{
  unsigned __int64 v2; // r8
  __int64 v3; // r13
  __int64 v4; // r13
  __int64 v5; // rbx
  _QWORD *v6; // rdi
  void (__fastcall *v7)(unsigned __int64, unsigned __int64, __int64); // rax

  v2 = *(_QWORD *)(a1 + 48);
  *(_QWORD *)a1 = &unk_4A0E878;
  if ( *(_DWORD *)(a1 + 60) )
  {
    v3 = *(unsigned int *)(a1 + 56);
    if ( (_DWORD)v3 )
    {
      v4 = 8 * v3;
      v5 = 0;
      do
      {
        v6 = *(_QWORD **)(v2 + v5);
        if ( v6 != (_QWORD *)-8LL && v6 )
        {
          sub_C7D6A0((__int64)v6, *v6 + 9LL, 8);
          v2 = *(_QWORD *)(a1 + 48);
        }
        v5 += 8;
      }
      while ( v4 != v5 );
    }
  }
  _libc_free(v2);
  v7 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(a1 + 32);
  if ( v7 )
    v7(a1 + 16, a1 + 16, 3);
  j_j___libc_free_0(a1);
}
