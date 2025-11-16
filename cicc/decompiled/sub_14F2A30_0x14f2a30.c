// Function: sub_14F2A30
// Address: 0x14f2a30
//
__int64 __fastcall sub_14F2A30(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v4; // rbx
  __int64 v5; // rdi
  __int64 result; // rax

  v2 = *(_QWORD *)(a1 + 8);
  if ( v2 != a2 )
  {
    v4 = a2;
    do
    {
      v5 = *(_QWORD *)(v4 + 16);
      if ( v5 )
        result = j_j___libc_free_0(v5, *(_QWORD *)(v4 + 32) - v5);
      v4 += 40;
    }
    while ( v2 != v4 );
    *(_QWORD *)(a1 + 8) = a2;
  }
  return result;
}
