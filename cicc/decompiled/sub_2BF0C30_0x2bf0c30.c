// Function: sub_2BF0C30
// Address: 0x2bf0c30
//
__int64 __fastcall sub_2BF0C30(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // r12
  __int64 v6; // rax

  v2 = sub_22077B0(0x88u);
  v5 = v2;
  if ( v2 )
    sub_2BEFCC0(v2, a2);
  v6 = *(unsigned int *)(a1 + 600);
  if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 604) )
  {
    sub_C8D5F0(a1 + 592, (const void *)(a1 + 608), v6 + 1, 8u, v3, v4);
    v6 = *(unsigned int *)(a1 + 600);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 592) + 8 * v6) = v5;
  ++*(_DWORD *)(a1 + 600);
  return v5;
}
