// Function: sub_2E0AFD0
// Address: 0x2e0afd0
//
void __fastcall sub_2E0AFD0(__int64 a1)
{
  unsigned __int64 *v1; // rbx
  unsigned __int64 *v2; // rsi

  v1 = *(unsigned __int64 **)(a1 + 104);
  while ( v1 )
  {
    v2 = v1;
    v1 = (unsigned __int64 *)v1[13];
    sub_2E0AED0(a1, v2);
  }
  *(_QWORD *)(a1 + 104) = 0;
}
