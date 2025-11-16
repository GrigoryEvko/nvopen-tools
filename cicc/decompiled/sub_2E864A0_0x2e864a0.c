// Function: sub_2E864A0
// Address: 0x2e864a0
//
unsigned __int64 __fastcall sub_2E864A0(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int64 v2; // rdx
  int v3; // eax

  v1 = *(_QWORD *)(a1 + 48);
  v2 = v1 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v1 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    return 0;
  v3 = v1 & 7;
  if ( !v3 )
  {
    *(_QWORD *)(a1 + 48) = v2;
    return a1 + 48;
  }
  if ( v3 == 3 )
    return v2 + 16;
  else
    return 0;
}
