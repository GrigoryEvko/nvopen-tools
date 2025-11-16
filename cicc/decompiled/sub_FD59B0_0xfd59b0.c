// Function: sub_FD59B0
// Address: 0xfd59b0
//
__int64 __fastcall sub_FD59B0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rbx
  __int64 v4; // rdi
  int v5; // eax
  int v6; // edx
  unsigned __int64 *v7; // rcx
  unsigned __int64 v8; // rdx
  _QWORD *v9; // r15
  _QWORD *v10; // r12
  __int64 v11; // rax
  _QWORD *v12; // rdi
  __int64 result; // rax

  v3 = (_QWORD *)a2;
  v4 = *(_QWORD *)(a2 + 16);
  if ( v4 )
  {
    v5 = *(_DWORD *)(v4 + 64);
    v6 = (v5 + 0x7FFFFFF) & 0x7FFFFFF;
    *(_DWORD *)(v4 + 64) = v6 | v5 & 0xF8000000;
    if ( !v6 )
    {
      a2 = a1;
      sub_FD59A0(v4, a1);
    }
    v3[2] = 0;
  }
  else
  {
    *(_DWORD *)(a1 + 56) -= *(_DWORD *)(a2 + 32);
  }
  v7 = (unsigned __int64 *)v3[1];
  v8 = *v3 & 0xFFFFFFFFFFFFFFF8LL;
  *v7 = v8 | *v7 & 7;
  *(_QWORD *)(v8 + 8) = v7;
  v9 = (_QWORD *)v3[6];
  v10 = (_QWORD *)v3[5];
  *v3 &= 7uLL;
  v3[1] = 0;
  if ( v9 != v10 )
  {
    do
    {
      v11 = v10[2];
      if ( v11 != 0 && v11 != -4096 && v11 != -8192 )
        sub_BD60C0(v10);
      v10 += 3;
    }
    while ( v9 != v10 );
    v10 = (_QWORD *)v3[5];
  }
  if ( v10 )
  {
    a2 = v3[7] - (_QWORD)v10;
    j_j___libc_free_0(v10, a2);
  }
  v12 = (_QWORD *)v3[3];
  if ( v3 + 5 != v12 )
    _libc_free(v12, a2);
  result = j_j___libc_free_0(v3, 72);
  if ( *(_QWORD **)(a1 + 64) == v3 )
    *(_QWORD *)(a1 + 64) = 0;
  return result;
}
