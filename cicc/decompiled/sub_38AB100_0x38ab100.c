// Function: sub_38AB100
// Address: 0x38ab100
//
__int64 __fastcall sub_38AB100(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  _BYTE *v9; // rsi
  __int64 v10; // rdx
  unsigned __int64 v11; // r14
  unsigned int v12; // r12d
  double v14; // xmm4_8
  double v15; // xmm5_8
  char v16; // [rsp+Ah] [rbp-56h] BYREF
  char v17; // [rsp+Bh] [rbp-55h] BYREF
  int v18; // [rsp+Ch] [rbp-54h] BYREF
  int v19; // [rsp+10h] [rbp-50h] BYREF
  int v20; // [rsp+14h] [rbp-4Ch] BYREF
  int v21; // [rsp+18h] [rbp-48h] BYREF
  int v22; // [rsp+1Ch] [rbp-44h] BYREF
  _BYTE *v23[2]; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v24[6]; // [rsp+30h] [rbp-30h] BYREF

  v9 = *(_BYTE **)(a1 + 72);
  v10 = *(_QWORD *)(a1 + 80);
  v23[0] = v24;
  v11 = *(_QWORD *)(a1 + 56);
  sub_3887850((__int64 *)v23, v9, (__int64)&v9[v10]);
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 3, "expected '=' in global variable")
    || (unsigned __int8)sub_388C1F0(a1, (__int64)&v18, &v16, &v19, &v20, &v17)
    || (unsigned __int8)sub_388BEC0(a1, &v21)
    || (unsigned __int8)sub_388ADB0(a1, &v22) )
  {
    v12 = 1;
  }
  else if ( (unsigned int)(*(_DWORD *)(a1 + 64) - 91) > 1 )
  {
    v12 = sub_38AA580(a1, v23, v11, v18, v16, v19, a2, a3, a4, a5, v14, v15, a8, a9, v20, v17, v21, v22);
  }
  else
  {
    v12 = sub_389EEE0(a1, (__int64 *)v23, v11, v18, v19, v20, a2, a3, a4, a5, v14, v15, a8, a9, v17, v21, v22);
  }
  if ( (_QWORD *)v23[0] != v24 )
    j_j___libc_free_0((unsigned __int64)v23[0]);
  return v12;
}
