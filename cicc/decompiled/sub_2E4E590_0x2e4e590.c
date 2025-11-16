// Function: sub_2E4E590
// Address: 0x2e4e590
//
__int64 __fastcall sub_2E4E590(__int64 a1, __int64 a2)
{
  char v2; // r14
  __int64 (*v3)(void); // rdx
  __int64 v4; // rax
  __int64 i; // rbx
  __int64 (*v7)(void); // rax

  if ( (_DWORD)qword_501F428 )
  {
    v2 = (_DWORD)qword_501F428 == 1;
  }
  else
  {
    v2 = 0;
    v7 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 472LL);
    if ( v7 != sub_2E44810 )
      v2 = v7();
  }
  *(_BYTE *)(a1 + 240) = 0;
  *(_QWORD *)a1 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  v3 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 128LL);
  v4 = 0;
  if ( v3 != sub_2DAC790 )
    v4 = v3();
  *(_QWORD *)(a1 + 8) = v4;
  *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 32);
  for ( i = *(_QWORD *)(a2 + 328); a2 + 320 != i; i = *(_QWORD *)(i + 8) )
  {
    if ( v2 )
      sub_2E48C90(a1, i);
    sub_2E4CB40(a1, i);
    sub_2E4B560(a1, i);
  }
  return *(unsigned __int8 *)(a1 + 240);
}
