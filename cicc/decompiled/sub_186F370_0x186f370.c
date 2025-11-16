// Function: sub_186F370
// Address: 0x186f370
//
bool __fastcall sub_186F370(__int64 *a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned __int8 *v3; // rax
  __int64 v4; // r12
  size_t v5; // rdx
  int v6; // eax
  __int64 v7; // rax

  v2 = *a1;
  v3 = (unsigned __int8 *)sub_1649960(a2);
  v4 = *(_QWORD *)v2 + 8LL * *(unsigned int *)(v2 + 8);
  v6 = sub_16D1B30((__int64 *)v2, v3, v5);
  if ( v6 == -1 )
    v7 = *(_QWORD *)v2 + 8LL * *(unsigned int *)(v2 + 8);
  else
    v7 = *(_QWORD *)v2 + 8LL * v6;
  return v7 != v4;
}
