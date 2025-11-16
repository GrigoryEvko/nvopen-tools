// Function: sub_B43E90
// Address: 0xb43e90
//
__int64 __fastcall sub_B43E90(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rcx
  __int64 v5; // rdi
  __int64 *v6; // r12
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 result; // rax
  __int64 v10; // rax
  __int64 v11; // rdx

  if ( !a2 )
    BUG();
  v3 = a1 + 24;
  v4 = *(_QWORD *)(a2 + 16);
  v5 = v4 + 48;
  if ( v4 + 48 == (*(_QWORD *)(v4 + 48) & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v6 = *(__int64 **)(v4 + 56);
    sub_AA48C0(v5, a1);
    v10 = *(_QWORD *)(a1 + 24);
    v11 = *v6;
    *(_QWORD *)(a1 + 32) = v6;
    v11 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a1 + 24) = v11 | v10 & 7;
    *(_QWORD *)(v11 + 8) = v3;
  }
  else
  {
    v6 = *(__int64 **)(a2 + 8);
    sub_AA48C0(v5, a1);
    v7 = *(_QWORD *)(a1 + 24);
    v8 = *v6;
    *(_QWORD *)(a1 + 32) = v6;
    v8 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a1 + 24) = v8 | v7 & 7;
    *(_QWORD *)(v8 + 8) = v3;
  }
  result = v3 | *v6 & 7;
  *v6 = result;
  return result;
}
