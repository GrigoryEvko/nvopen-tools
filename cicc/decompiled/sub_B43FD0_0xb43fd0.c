// Function: sub_B43FD0
// Address: 0xb43fd0
//
__int64 __fastcall sub_B43FD0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v3; // [rsp+10h] [rbp-10h]

  v1 = sub_AA6190(*(_QWORD *)(a1 + 40), a1);
  if ( !v1 || v1 + 8 == (*(_QWORD *)(v1 + 8) & 0xFFFFFFFFFFFFFFF8LL) )
    return v3;
  else
    return *(_QWORD *)(v1 + 16);
}
