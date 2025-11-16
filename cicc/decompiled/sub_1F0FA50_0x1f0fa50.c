// Function: sub_1F0FA50
// Address: 0x1f0fa50
//
__int64 __fastcall sub_1F0FA50(__int64 a1, __int64 *a2)
{
  __int64 v4; // rsi
  __int64 v5; // rcx
  __int64 v6; // rdi
  __int64 v7; // r8
  __int64 (*v8)(); // rax
  __int64 (*v9)(); // rax
  __int64 v11; // rax
  int v12; // eax

  v4 = *a2;
  if ( (unsigned __int8)sub_1636880(a1, v4) || (__int64 *)(a2[40] & 0xFFFFFFFFFFFFFFF8LL) == a2 + 40 )
    return 0;
  v6 = a2[2];
  v7 = 0;
  v8 = *(__int64 (**)())(*(_QWORD *)v6 + 48LL);
  if ( v8 != sub_1D90020 )
    v7 = ((__int64 (__fastcall *)(__int64, __int64, __int64 *, __int64, _QWORD))v8)(v6, v4, a2 + 40, v5, 0);
  if ( dword_4FCA800 != 1 )
  {
    if ( dword_4FCA800 == 2 )
      return 0;
    v9 = *(__int64 (**)())(*(_QWORD *)v7 + 56LL);
    if ( v9 == sub_1F0BF20 )
      return 0;
    if ( !((unsigned __int8 (__fastcall *)(__int64, __int64 *))v9)(v7, a2) )
      return 0;
    v11 = *(_QWORD *)(a2[1] + 608);
    if ( *(_DWORD *)(v11 + 348) == 4 )
    {
      v12 = *(_DWORD *)(v11 + 352);
      if ( v12 )
      {
        if ( v12 != 6 )
          return 0;
      }
    }
    if ( (unsigned __int8)sub_1560180(*a2 + 112, 42)
      || (unsigned __int8)sub_1560180(*a2 + 112, 45)
      || (unsigned __int8)sub_1560180(*a2 + 112, 44)
      || (unsigned __int8)sub_1560180(*a2 + 112, 43) )
    {
      return 0;
    }
  }
  sub_1F0EBC0(a1, a2);
  return 0;
}
