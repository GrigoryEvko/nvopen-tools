// Function: sub_1888BA0
// Address: 0x1888ba0
//
__int64 __fastcall sub_1888BA0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // r13
  __int64 v4; // r15
  __int64 v5; // r14
  unsigned __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v13; // [rsp+0h] [rbp-40h]

  v3 = 0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 4);
  v13 = a2 - a1;
  if ( a2 - a1 <= 0 )
    return a3;
  v4 = a3 - 40;
  v5 = a2 - 40;
  v6 = 0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 4);
  do
  {
    v7 = *(_QWORD *)(v4 + 8);
    while ( v7 )
    {
      sub_1876060(*(_QWORD *)(v7 + 24));
      v8 = v7;
      v7 = *(_QWORD *)(v7 + 16);
      j_j___libc_free_0(v8, 40);
    }
    *(_QWORD *)(v4 + 8) = 0;
    *(_QWORD *)(v4 + 16) = v4;
    *(_QWORD *)(v4 + 24) = v4;
    *(_QWORD *)(v4 + 32) = 0;
    if ( *(_QWORD *)(v5 + 8) )
    {
      *(_DWORD *)v4 = *(_DWORD *)v5;
      v9 = *(_QWORD *)(v5 + 8);
      *(_QWORD *)(v4 + 8) = v9;
      *(_QWORD *)(v4 + 16) = *(_QWORD *)(v5 + 16);
      *(_QWORD *)(v4 + 24) = *(_QWORD *)(v5 + 24);
      *(_QWORD *)(v9 + 8) = v4;
      *(_QWORD *)(v4 + 32) = *(_QWORD *)(v5 + 32);
      *(_QWORD *)(v5 + 8) = 0;
      *(_QWORD *)(v5 + 16) = v5;
      *(_QWORD *)(v5 + 24) = v5;
      *(_QWORD *)(v5 + 32) = 0;
    }
    v4 -= 48;
    v5 -= 48;
    --v6;
  }
  while ( v6 );
  v10 = -48;
  v11 = -48LL * v3;
  if ( v13 > 0 )
    v10 = v11;
  return a3 + v10;
}
