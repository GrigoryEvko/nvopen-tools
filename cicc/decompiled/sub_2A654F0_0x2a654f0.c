// Function: sub_2A654F0
// Address: 0x2a654f0
//
char __fastcall sub_2A654F0(_QWORD *a1)
{
  __int64 v1; // rax
  __int64 *v2; // rbx
  __int64 v3; // rax
  __int64 *i; // r12
  __int64 v5; // rdi
  __int64 v6; // rdx

  v1 = sub_2A654E0(a1);
  v2 = *(__int64 **)(v1 + 32);
  v3 = *(unsigned int *)(v1 + 40);
  for ( i = &v2[6 * v3]; i != v2; LOBYTE(v3) = sub_2A61DD0(v5, 0, v6) )
  {
    v5 = *v2;
    v6 = (__int64)(v2 + 1);
    v2 += 6;
  }
  return v3;
}
