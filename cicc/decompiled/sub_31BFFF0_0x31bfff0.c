// Function: sub_31BFFF0
// Address: 0x31bfff0
//
__int64 __fastcall sub_31BFFF0(__int64 **a1)
{
  __int64 v1; // r12
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 v6; // r13

  v1 = **a1;
  v2 = sub_31BFFE0(a1, 1);
  if ( v2 != v3 )
  {
    v4 = v2;
    v5 = v3;
    do
    {
      v6 = *(_QWORD *)v4;
      if ( sub_B445A0(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)v4 + 8LL) + 16LL), *(_QWORD *)(*(_QWORD *)(v1 + 8) + 16LL)) )
        v1 = v6;
      v4 += 8;
    }
    while ( v5 != v4 );
  }
  return v1;
}
