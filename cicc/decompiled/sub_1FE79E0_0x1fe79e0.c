// Function: sub_1FE79E0
// Address: 0x1fe79e0
//
__int64 __fastcall sub_1FE79E0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 *v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 (*v10)(); // rcx
  __int64 (*v11)(); // rcx
  __int64 v12; // rdx
  __int64 (*v13)(void); // rdx
  __int64 result; // rax

  v5 = *(_QWORD *)(a2 + 56);
  v6 = *(_QWORD *)(v5 + 40);
  *a1 = v5;
  a1[1] = v6;
  v7 = *(__int64 **)(v5 + 16);
  v8 = 0;
  v9 = *v7;
  v10 = *(__int64 (**)())(*v7 + 40);
  if ( v10 != sub_1D00B00 )
  {
    v8 = ((__int64 (__fastcall *)(__int64 *, __int64, _QWORD))v10)(v7, a2, 0);
    v7 = *(__int64 **)(*a1 + 16LL);
    v9 = *v7;
  }
  a1[2] = v8;
  v11 = *(__int64 (**)())(v9 + 112);
  v12 = 0;
  if ( v11 != sub_1D00B10 )
  {
    v12 = ((__int64 (__fastcall *)(__int64 *, __int64, _QWORD))v11)(v7, a2, 0);
    v9 = **(_QWORD **)(*a1 + 16LL);
  }
  a1[3] = v12;
  v13 = *(__int64 (**)(void))(v9 + 56);
  result = 0;
  if ( v13 != sub_1D12D20 )
    result = v13();
  a1[5] = a2;
  a1[6] = a3;
  a1[4] = result;
  return result;
}
