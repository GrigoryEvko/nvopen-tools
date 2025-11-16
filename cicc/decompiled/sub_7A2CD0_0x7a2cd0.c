// Function: sub_7A2CD0
// Address: 0x7a2cd0
//
__int64 __fastcall sub_7A2CD0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rax

  v2 = sub_72B840(a2);
  v6 = *(_QWORD *)(v2 + 80);
  if ( *(_BYTE *)(v6 + 40) == 19 )
    v6 = *(_QWORD *)(*(_QWORD *)(v6 + 72) + 8LL);
  return sub_7987E0(a1, *(_QWORD *)(v6 + 72), v2, v3, v4, v5);
}
