// Function: sub_1E1A330
// Address: 0x1e1a330
//
__int64 __fastcall sub_1E1A330(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 a5,
        __int64 a6,
        __int64 a7)
{
  unsigned __int8 v10; // bl
  __int64 v11; // rax
  __int64 *v12; // rax
  __int64 v13; // r9
  bool v14; // r8
  __int64 v15; // rsi
  char v16; // r8
  __int64 v18; // rdi
  __int64 (*v19)(); // rax
  __int64 v20; // rax
  __int64 v21; // [rsp+8h] [rbp-78h]
  __int64 v22; // [rsp+8h] [rbp-78h]
  char v23; // [rsp+10h] [rbp-70h]
  unsigned __int8 v24; // [rsp+1Ch] [rbp-64h]
  __int64 v25[12]; // [rsp+20h] [rbp-60h] BYREF

  v10 = a3;
  v11 = *(_QWORD *)(a1 + 24);
  v24 = a4;
  if ( v11 && (v12 = *(__int64 **)(v11 + 56)) != 0 )
  {
    v13 = *v12;
    v14 = 1;
    v15 = *(_QWORD *)(*v12 + 40);
    if ( !a7 )
    {
      v18 = v12[2];
      v14 = 0;
      v19 = *(__int64 (**)())(*(_QWORD *)v18 + 40LL);
      if ( v19 != sub_1D00B00 )
      {
        v22 = v13;
        v20 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, _QWORD))v19)(v18, v15, a3, a4, 0);
        v13 = v22;
        v14 = v20 != 0;
      }
    }
    v23 = v14;
    v21 = v13;
    sub_154BA10((__int64)v25, v15, 1);
    sub_154C150((__int64)v25, v21);
    v16 = v23;
  }
  else
  {
    sub_154BA10((__int64)v25, 0, 1);
    v16 = a7 != 0;
  }
  sub_1E181D0(a1, a2, (__int64)v25, v10, v24, (_BYTE *)a5, v16, 0);
  return sub_154BA40(v25);
}
