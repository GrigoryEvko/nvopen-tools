// Function: sub_321F7C0
// Address: 0x321f7c0
//
__int64 __fastcall sub_321F7C0(__int64 a1, __int64 a2)
{
  __int64 (*v2)(); // rax

  v2 = *(__int64 (**)())(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 104LL);
  if ( v2 == sub_C13EF0 || !(unsigned __int8)v2() )
    return *(unsigned int *)(a2 + 72);
  else
    return 0;
}
