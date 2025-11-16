// Function: sub_16368E0
// Address: 0x16368e0
//
__int64 __fastcall sub_16368E0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 (*v6)(); // rax

  v2 = *(_QWORD *)(a2 + 56);
  if ( !v2 )
    return 0;
  v4 = sub_15E0530(*(_QWORD *)(a2 + 56));
  v5 = sub_16033D0(v4);
  v6 = *(__int64 (**)())(*(_QWORD *)v5 + 32LL);
  if ( v6 == sub_1635EE0 || ((unsigned __int8 (__fastcall *)(__int64, __int64, __int64))v6)(v5, a1, a2) )
    return sub_1560180(v2 + 112, 35);
  else
    return 1;
}
