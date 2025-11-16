// Function: sub_2EA49A0
// Address: 0x2ea49a0
//
__int64 __fastcall sub_2EA49A0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // rdx

  v1 = sub_2EA48E0(a1);
  if ( !v1 )
    return 0;
  v2 = v1;
  if ( !(unsigned __int8)sub_2E31B00(v1) )
    return 0;
  v3 = *(_QWORD *)(v2 + 112);
  v4 = v3 + 8LL * *(unsigned int *)(v2 + 120);
  if ( v3 == v4 || v4 != v3 + 8 )
    return 0;
  return v2;
}
