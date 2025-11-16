// Function: sub_2E5ED20
// Address: 0x2e5ed20
//
__int64 __fastcall sub_2E5ED20(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_2E5EC00(a1);
  if ( v1 && (v2 = v1, *(_DWORD *)(v1 + 120) == 1) && (unsigned __int8)sub_2E31B00(v1) )
    return v2;
  else
    return 0;
}
