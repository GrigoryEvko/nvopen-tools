// Function: sub_C4F890
// Address: 0xc4f890
//
__int64 __fastcall sub_C4F890(__int64 a1, __int64 a2)
{
  __int64 v3; // r8
  __int64 v4; // r13
  __int64 v5; // r13
  __int64 v6; // rbx
  _QWORD *v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 result; // rax

  if ( a1 )
  {
    v3 = *(_QWORD *)(a1 + 128);
    if ( *(_DWORD *)(a1 + 140) )
    {
      v4 = *(unsigned int *)(a1 + 136);
      if ( (_DWORD)v4 )
      {
        v5 = 8 * v4;
        v6 = 0;
        do
        {
          v7 = *(_QWORD **)(v3 + v6);
          if ( v7 != (_QWORD *)-8LL && v7 )
          {
            a2 = *v7 + 17LL;
            sub_C7D6A0(v7, a2, 8);
            v3 = *(_QWORD *)(a1 + 128);
          }
          v6 += 8;
        }
        while ( v5 != v6 );
      }
    }
    _libc_free(v3, a2);
    v8 = *(_QWORD *)(a1 + 80);
    if ( v8 != a1 + 96 )
      _libc_free(v8, a2);
    v9 = *(_QWORD *)(a1 + 32);
    if ( v9 != a1 + 48 )
      _libc_free(v9, a2);
    return j_j___libc_free_0(a1, 160);
  }
  return result;
}
