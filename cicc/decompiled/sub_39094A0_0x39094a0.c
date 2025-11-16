// Function: sub_39094A0
// Address: 0x39094a0
//
__int64 __fastcall sub_39094A0(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // r13
  __int64 v5; // rsi

  if ( *(_DWORD *)sub_3909460(a1) == 1 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 136LL))(a1);
  v3 = *(_QWORD *)(a1 + 24);
  v4 = v3 + 104LL * *(unsigned int *)(a1 + 32);
  while ( v3 != v4 )
  {
    v5 = v3 + 8;
    v3 += 104;
    sub_16E2F40(a2, v5);
  }
  return 1;
}
