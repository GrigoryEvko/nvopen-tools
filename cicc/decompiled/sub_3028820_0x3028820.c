// Function: sub_3028820
// Address: 0x3028820
//
void __fastcall sub_3028820(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rsi
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // r15

  v5 = *(unsigned int *)(a4 + 4);
  if ( (_DWORD)v5 )
  {
    v7 = (*(__int64 (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 480))(a1, v5, 0);
    if ( v7 )
    {
      v8 = sub_A777F0(0x10u, a1 + 123);
      v9 = v8;
      if ( v8 )
      {
        *(_QWORD *)v8 = 0;
        *(_DWORD *)(v8 + 8) = 0;
      }
      sub_3249A20(a2, v8, 0, 65547, 144);
      sub_3249A20(a2, v9, 0, 65551, v7);
      sub_32498C0(a2, a3, 2, v9);
      sub_3249A20(a2, a3 + 8, 51, 65547, 2);
    }
  }
}
