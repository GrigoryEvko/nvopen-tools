// Function: sub_2AB1A90
// Address: 0x2ab1a90
//
__int64 __fastcall sub_2AB1A90(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // r15
  __int64 v8; // r13
  __int64 v9; // rdi

  v3 = sub_2BF9BD0();
  v4 = a1 + 112;
  v5 = sub_2BF0CC0(v3, a2);
  v6 = *(_QWORD *)(a1 + 120);
  v7 = v5;
  v8 = v5 + 112;
  if ( a1 + 112 != v6 )
  {
    do
    {
      v9 = v6;
      v6 = *(_QWORD *)(v6 + 8);
      sub_2C19EE0(v9 - 24, v7, v8);
    }
    while ( v4 != v6 );
  }
  return sub_2AB1780(a1, v7);
}
