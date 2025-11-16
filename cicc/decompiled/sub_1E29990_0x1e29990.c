// Function: sub_1E29990
// Address: 0x1e29990
//
__int64 __fastcall sub_1E29990(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_1E29910(a1);
  if ( v1 && (v2 = v1, (unsigned __int8)sub_1DD61E0(v1)) && *(_QWORD *)(v2 + 96) == *(_QWORD *)(v2 + 88) + 8LL )
    return v2;
  else
    return 0;
}
