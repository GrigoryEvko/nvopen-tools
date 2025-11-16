// Function: sub_1404700
// Address: 0x1404700
//
__int64 __fastcall sub_1404700(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 (*v7)(); // rax

  v2 = **(_QWORD **)(a2 + 32);
  v3 = *(_QWORD *)(v2 + 56);
  if ( !v3 )
    return 0;
  v5 = sub_15E0530(*(_QWORD *)(v2 + 56));
  v6 = sub_16033D0(v5);
  v7 = *(__int64 (**)())(*(_QWORD *)v6 + 48LL);
  if ( v7 == sub_1403590 || ((unsigned __int8 (__fastcall *)(__int64, __int64, __int64))v7)(v6, a1, a2) )
    return sub_1560180(v3 + 112, 35);
  else
    return 1;
}
