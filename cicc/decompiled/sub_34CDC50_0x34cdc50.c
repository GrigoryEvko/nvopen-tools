// Function: sub_34CDC50
// Address: 0x34cdc50
//
__int64 __fastcall sub_34CDC50(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r8
  __int64 v4; // rdi
  __int64 (*v5)(); // rcx

  v3 = 0;
  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL);
  v5 = *(__int64 (**)())(*(_QWORD *)v4 + 96LL);
  if ( v5 != sub_23CE310 )
    return ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64 (*)(), _QWORD))v5)(v4, a2, a3, v5, 0);
  return v3;
}
