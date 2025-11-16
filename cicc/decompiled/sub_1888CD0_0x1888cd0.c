// Function: sub_1888CD0
// Address: 0x1888cd0
//
__int64 __fastcall sub_1888CD0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  unsigned __int64 v4; // r13
  __int64 v5; // r12
  __int64 v6; // r15
  __int64 v7; // rbx
  __int64 v8; // rdi
  __int64 v9; // rdx

  v3 = a2 - a1;
  v4 = 0xAAAAAAAAAAAAAAABLL * (v3 >> 4);
  if ( v3 <= 0 )
    return a3;
  v5 = a3 + 8;
  v6 = a1 + 8;
  do
  {
    v7 = *(_QWORD *)(v5 + 8);
    while ( v7 )
    {
      sub_1876060(*(_QWORD *)(v7 + 24));
      v8 = v7;
      v7 = *(_QWORD *)(v7 + 16);
      j_j___libc_free_0(v8, 40);
    }
    *(_QWORD *)(v5 + 8) = 0;
    *(_QWORD *)(v5 + 16) = v5;
    *(_QWORD *)(v5 + 24) = v5;
    *(_QWORD *)(v5 + 32) = 0;
    if ( *(_QWORD *)(v6 + 8) )
    {
      *(_DWORD *)v5 = *(_DWORD *)v6;
      v9 = *(_QWORD *)(v6 + 8);
      *(_QWORD *)(v5 + 8) = v9;
      *(_QWORD *)(v5 + 16) = *(_QWORD *)(v6 + 16);
      *(_QWORD *)(v5 + 24) = *(_QWORD *)(v6 + 24);
      *(_QWORD *)(v9 + 8) = v5;
      *(_QWORD *)(v5 + 32) = *(_QWORD *)(v6 + 32);
      *(_QWORD *)(v6 + 8) = 0;
      *(_QWORD *)(v6 + 16) = v6;
      *(_QWORD *)(v6 + 24) = v6;
      *(_QWORD *)(v6 + 32) = 0;
    }
    v5 += 48;
    v6 += 48;
    --v4;
  }
  while ( v4 );
  return v3 + a3;
}
