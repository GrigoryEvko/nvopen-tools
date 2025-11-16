// Function: sub_2E91850
// Address: 0x2e91850
//
__int64 (__fastcall *__fastcall sub_2E91850(
        __int64 a1,
        __int64 a2,
        unsigned __int8 a3,
        char a4,
        char a5,
        char a6,
        __int64 a7))(_QWORD *, _QWORD *, __int64)
{
  __int64 v10; // rax
  __int64 *v11; // rax
  __int64 v12; // r8
  __int64 v13; // rsi
  __int64 (*v15)(void); // rax
  __int64 v16; // rax
  __int64 v17; // [rsp+8h] [rbp-B8h]
  __int64 v18; // [rsp+10h] [rbp-B0h]
  _QWORD v21[20]; // [rsp+20h] [rbp-A0h] BYREF

  v10 = *(_QWORD *)(a1 + 24);
  if ( v10 && (v11 = *(__int64 **)(v10 + 32)) != 0 )
  {
    v12 = *v11;
    v13 = *(_QWORD *)(*v11 + 40);
    if ( !a7 )
    {
      v15 = *(__int64 (**)(void))(*(_QWORD *)v11[2] + 128LL);
      if ( v15 != sub_2DAC790 )
      {
        v17 = v12;
        v16 = v15();
        v12 = v17;
        a7 = v16;
      }
    }
    v18 = v12;
    sub_A558A0((__int64)v21, v13, 1);
    sub_A564B0((__int64)v21, v18);
  }
  else
  {
    sub_A558A0((__int64)v21, 0, 1);
  }
  sub_2E8BCA0(a1, a2, (__int64)v21, a3, a4, a5, a6, a7);
  return sub_A55520(v21, a2);
}
