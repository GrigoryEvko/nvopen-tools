// Function: sub_EAC8B0
// Address: 0xeac8b0
//
__int64 __fastcall sub_EAC8B0(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rax
  __int64 v4; // r15
  unsigned int v5; // eax
  unsigned int v6; // r13d
  __int64 v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // r8
  __int64 (*v10)(); // rax
  __int64 v12; // rax
  __int64 v13; // [rsp+8h] [rbp-78h]
  __int64 v14; // [rsp+18h] [rbp-68h] BYREF
  _QWORD v15[4]; // [rsp+20h] [rbp-60h] BYREF
  char v16; // [rsp+40h] [rbp-40h]
  char v17; // [rsp+41h] [rbp-3Fh]

  v3 = sub_ECD690(a1 + 40);
  v15[0] = 0;
  v4 = v3;
  LOBYTE(v5) = sub_EAC4D0(a1, &v14, (__int64)v15);
  v6 = v5;
  if ( !(_BYTE)v5 )
  {
    v7 = *(_QWORD *)(a1 + 232);
    v8 = 0;
    v9 = v14;
    v10 = *(__int64 (**)())(*(_QWORD *)v7 + 80LL);
    if ( v10 != sub_C13ED0 )
    {
      v13 = v14;
      v12 = ((__int64 (__fastcall *)(__int64, __int64 *, _QWORD))v10)(v7, &v14, 0);
      v9 = v13;
      v8 = v12;
    }
    if ( !sub_E81930(v9, a2, v8) )
    {
      v17 = 1;
      v15[0] = "expected absolute expression";
      v16 = 3;
      return (unsigned int)sub_ECDA70(a1, v4, v15, 0, 0);
    }
  }
  return v6;
}
