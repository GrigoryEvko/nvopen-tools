// Function: sub_38C4F40
// Address: 0x38c4f40
//
__int64 __fastcall sub_38C4F40(__int64 *a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // r13
  __int64 v5; // r15
  __int64 v6; // rbx

  v4 = a2;
  v5 = a1[1];
  if ( !*(_BYTE *)(*(_QWORD *)(v5 + 16) + 297LL) )
  {
    v6 = sub_38BFA60(a1[1], 1);
    (*(void (__fastcall **)(__int64 *, __int64, __int64))(*a1 + 240))(a1, v6, a2);
    v4 = sub_38CF310(v6, 0, v5, 0);
  }
  return sub_38DDD30(a1, v4, a3, 0);
}
