// Function: sub_35A5710
// Address: 0x35a5710
//
void (*__fastcall sub_35A5710(__int64 a1))()
{
  __int64 v2; // rax
  __int64 v3; // rsi
  void (__fastcall *v4)(__int64 *, __int64, _QWORD); // rcx
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v12; // [rsp+8h] [rbp-18h] BYREF

  *(_QWORD *)(a1 + 48) = sub_2EA6400(**(_QWORD **)a1);
  v2 = sub_2EA49A0(**(_QWORD **)a1);
  v3 = *(_QWORD *)(a1 + 32);
  *(_QWORD *)(a1 + 56) = v2;
  v4 = *(void (__fastcall **)(__int64 *, __int64, _QWORD))(*(_QWORD *)v3 + 376LL);
  v5 = 0;
  if ( (char *)v4 != (char *)sub_2FDC520 )
  {
    v4(&v12, v3, *(_QWORD *)(a1 + 48));
    v5 = v12;
  }
  v6 = *(_QWORD *)(a1 + 528);
  v12 = 0;
  *(_QWORD *)(a1 + 528) = v5;
  if ( v6 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL))(v6);
    if ( v12 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v12 + 8LL))(v12);
  }
  sub_35A10A0(a1);
  sub_35A4AE0((__int64 *)a1, v3, v7, v8, v9, v10);
  return sub_3599F20(a1);
}
