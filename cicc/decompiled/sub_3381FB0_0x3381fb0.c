// Function: sub_3381FB0
// Address: 0x3381fb0
//
void __fastcall sub_3381FB0(__int64 a1)
{
  __int64 v1; // r12
  __int64 i; // r14
  __int64 v3; // rbx
  __int64 v4; // r13
  __int64 v5; // rdx

  v1 = *(_QWORD *)(a1 + 104);
  for ( i = v1 + 32LL * *(unsigned int *)(a1 + 112); i != v1; v1 += 32 )
  {
    v3 = *(_QWORD *)(v1 + 8);
    v4 = *(_QWORD *)(v1 + 16);
    while ( v4 != v3 )
    {
      v5 = v3;
      v3 += 32;
      sub_3381C20(a1, *(_BYTE **)v1, v5);
    }
  }
  sub_3373460(a1);
}
