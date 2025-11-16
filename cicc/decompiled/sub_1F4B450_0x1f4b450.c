// Function: sub_1F4B450
// Address: 0x1f4b450
//
char __fastcall sub_1F4B450(__int64 a1, __int64 a2)
{
  __int64 (*v2)(); // rax
  __int64 v3; // rbx
  __int64 v4; // rax
  _QWORD *v5; // r14
  __int64 (__fastcall *v6)(__int64, __int64); // rax
  char result; // al

  v2 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 48LL);
  if ( v2 == sub_1D90020 )
    BUG();
  v3 = *(_QWORD *)(a2 + 56);
  v4 = v2();
  v5 = (_QWORD *)(*(_QWORD *)a2 + 112LL);
  if ( *(_DWORD *)(v4 + 12) < *(_DWORD *)(v3 + 60) || (unsigned __int8)sub_1560180(*(_QWORD *)a2 + 112LL, 48) )
  {
    sub_15602E0(v5, "stackrealign", 0xCu);
  }
  else
  {
    result = sub_15602E0(v5, "stackrealign", 0xCu);
    if ( !result )
      return result;
  }
  v6 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 336LL);
  if ( v6 == sub_1F49D00 )
    return !sub_15602E0((_QWORD *)(*(_QWORD *)a2 + 112LL), "no-realign-stack", 0x10u);
  else
    return v6(a1, a2);
}
