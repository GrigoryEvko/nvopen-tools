// Function: sub_7E9190
// Address: 0x7e9190
//
void __fastcall sub_7E9190(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rdi

  v2 = *(_QWORD *)(a1 + 24);
  while ( v2 )
  {
    v3 = v2;
    v2 = *(_QWORD *)(v2 + 32);
    sub_7E90E0(v3, a2);
  }
}
