// Function: sub_1AEC950
// Address: 0x1aec950
//
void __fastcall sub_1AEC950(__int64 a1, __int64 a2, __int64 a3, double a4, double a5, double a6)
{
  __int64 **v7; // r13
  char v8; // al
  unsigned __int64 v9; // rax
  __int64 *v10; // r14
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15[5]; // [rsp+8h] [rbp-28h] BYREF

  v7 = *(__int64 ***)a3;
  v8 = *(_BYTE *)(*(_QWORD *)a3 + 8LL);
  if ( v8 == 15 )
  {
    sub_1625C10(a3, 11, a2);
  }
  else if ( v8 == 11 )
  {
    v15[0] = sub_16498A0(a3);
    v9 = sub_1599A20(**(__int64 ****)(a1 - 24));
    v10 = (__int64 *)sub_15A4180(v9, v7, 0);
    v11 = sub_159C470((__int64)v7, 1, 0);
    v12 = sub_15A2B30(v10, v11, 0, 0, a4, a5, a6);
    v14 = sub_161C350(v15, v12, (__int64)v10, v13);
    sub_1625C10(a3, 4, v14);
  }
}
