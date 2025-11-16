// Function: sub_38EB9C0
// Address: 0x38eb9c0
//
__int64 __fastcall sub_38EB9C0(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rax
  __int64 v4; // r15
  unsigned int v5; // eax
  unsigned int v6; // r13d
  __int64 v7; // rdi
  int v8; // edx
  __int64 v9; // r8
  __int64 (*v10)(); // rax
  int v12; // eax
  __int64 v13; // [rsp+8h] [rbp-68h]
  __int64 v14; // [rsp+18h] [rbp-58h] BYREF
  _QWORD v15[2]; // [rsp+20h] [rbp-50h] BYREF
  char v16; // [rsp+30h] [rbp-40h]
  char v17; // [rsp+31h] [rbp-3Fh]

  v3 = sub_3909290(a1 + 144);
  v15[0] = 0;
  v4 = v3;
  LOBYTE(v5) = sub_38EB6A0(a1, &v14, (__int64)v15);
  v6 = v5;
  if ( !(_BYTE)v5 )
  {
    v7 = *(_QWORD *)(a1 + 328);
    v8 = 0;
    v9 = v14;
    v10 = *(__int64 (**)())(*(_QWORD *)v7 + 72LL);
    if ( v10 != sub_168DB40 )
    {
      v13 = v14;
      v12 = ((__int64 (__fastcall *)(__int64, __int64 *, _QWORD))v10)(v7, &v14, 0);
      v9 = v13;
      v8 = v12;
    }
    if ( !sub_38CF2B0(v9, a2, v8) )
    {
      v17 = 1;
      v15[0] = "expected absolute expression";
      v16 = 3;
      return (unsigned int)sub_3909790(a1, v4, v15, 0, 0);
    }
  }
  return v6;
}
