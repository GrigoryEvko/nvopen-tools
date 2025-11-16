// Function: sub_2D57F20
// Address: 0x2d57f20
//
__int64 __fastcall sub_2D57F20(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  int v4; // edx
  unsigned __int64 v5; // r14
  __int64 v6; // r15
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // r13
  unsigned __int64 v9; // rdi
  __int64 v10; // [rsp-40h] [rbp-40h]

  result = *(_QWORD *)(a1 + 824);
  if ( !result )
  {
    result = sub_22077B0(0x80u);
    if ( result )
    {
      *(_QWORD *)(result + 96) = 0;
      *(_QWORD *)result = result + 16;
      *(_QWORD *)(result + 24) = result + 40;
      v4 = *(_DWORD *)(a2 + 92);
      *(_QWORD *)(result + 8) = 0x100000000LL;
      *(_QWORD *)(result + 32) = 0x600000000LL;
      *(_BYTE *)(result + 112) = 0;
      *(_DWORD *)(result + 116) = 0;
      *(_QWORD *)(result + 104) = a2;
      *(_DWORD *)(result + 120) = v4;
      v10 = result;
      sub_B1F440(result);
      result = v10;
    }
    v5 = *(_QWORD *)(a1 + 824);
    *(_QWORD *)(a1 + 824) = result;
    if ( v5 )
    {
      v6 = *(_QWORD *)(v5 + 24);
      v7 = v6 + 8LL * *(unsigned int *)(v5 + 32);
      if ( v6 != v7 )
      {
        do
        {
          v8 = *(_QWORD *)(v7 - 8);
          v7 -= 8LL;
          if ( v8 )
          {
            v9 = *(_QWORD *)(v8 + 24);
            if ( v9 != v8 + 40 )
              _libc_free(v9);
            j_j___libc_free_0(v8);
          }
        }
        while ( v6 != v7 );
        v7 = *(_QWORD *)(v5 + 24);
      }
      if ( v7 != v5 + 40 )
        _libc_free(v7);
      if ( *(_QWORD *)v5 != v5 + 16 )
        _libc_free(*(_QWORD *)v5);
      j_j___libc_free_0(v5);
      return *(_QWORD *)(a1 + 824);
    }
  }
  return result;
}
