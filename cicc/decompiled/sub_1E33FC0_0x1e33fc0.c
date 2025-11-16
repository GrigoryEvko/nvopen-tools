// Function: sub_1E33FC0
// Address: 0x1e33fc0
//
__int64 __fastcall sub_1E33FC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // rdi
  __int64 (*v13)(); // rax
  __int64 v14; // rdi
  __int64 (*v15)(); // rax
  __int64 v17; // [rsp+8h] [rbp-68h]
  __int64 v18[12]; // [rsp+10h] [rbp-60h] BYREF

  v8 = a5;
  v9 = *(_QWORD *)(a1 + 16);
  if ( v9 )
  {
    v10 = *(_QWORD *)(v9 + 24);
    if ( v10 )
    {
      v11 = *(_QWORD *)(v10 + 56);
      if ( v11 )
      {
        v12 = *(_QWORD *)(v11 + 16);
        a4 = 0;
        v13 = *(__int64 (**)())(*(_QWORD *)v12 + 112LL);
        if ( v13 != sub_1D00B10 )
          a4 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64))v13)(v12, a2, a3, a5);
        v14 = *(_QWORD *)(v11 + 8);
        v8 = 0;
        v15 = *(__int64 (**)())(*(_QWORD *)v14 + 32LL);
        if ( v15 != sub_16FF770 )
          v8 = ((__int64 (__fastcall *)(__int64, __int64, __int64, _QWORD))v15)(v14, a2, a3, 0);
      }
    }
  }
  v17 = v8;
  sub_154BA10((__int64)v18, 0, 1);
  sub_1E32250(a1, a2, (__int64)v18, a3, 0, 1, 1, 0, a4, v17);
  return sub_154BA40(v18);
}
