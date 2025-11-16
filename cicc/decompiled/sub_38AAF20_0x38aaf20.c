// Function: sub_38AAF20
// Address: 0x38aaf20
//
__int64 __fastcall sub_38AAF20(
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
  unsigned __int64 v10; // r13
  bool v11; // zf
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // rax
  unsigned int v16; // r12d
  double v18; // xmm4_8
  double v19; // xmm5_8
  char v20; // [rsp+6h] [rbp-AAh] BYREF
  char v21; // [rsp+7h] [rbp-A9h] BYREF
  int v22; // [rsp+8h] [rbp-A8h] BYREF
  int v23; // [rsp+Ch] [rbp-A4h] BYREF
  _QWORD v24[4]; // [rsp+10h] [rbp-A0h] BYREF
  _QWORD v25[2]; // [rsp+30h] [rbp-80h] BYREF
  __int16 v26; // [rsp+40h] [rbp-70h]
  _QWORD v27[2]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v28; // [rsp+60h] [rbp-50h]
  _BYTE *v29[2]; // [rsp+70h] [rbp-40h] BYREF
  _BYTE v30[48]; // [rsp+80h] [rbp-30h] BYREF

  v29[0] = v30;
  v10 = *(_QWORD *)(a1 + 56);
  v11 = *(_DWORD *)(a1 + 64) == 368;
  v29[1] = 0;
  v30[0] = 0;
  v12 = *(_QWORD *)(a1 + 1008);
  v13 = *(_QWORD *)(a1 + 1000);
  if ( v11 )
  {
    v14 = a1 + 8;
    v15 = (v12 - v13) >> 3;
    if ( (_DWORD)v15 != *(_DWORD *)(a1 + 104) )
    {
      LODWORD(v24[0]) = v15;
      v25[0] = "variable expected to be numbered '%";
      v28 = 770;
      v25[1] = v24[0];
      v26 = 2307;
      v27[0] = v25;
      v27[1] = "'";
      v16 = sub_38814C0(v14, v10, (__int64)v27);
      goto LABEL_7;
    }
    *(_DWORD *)(a1 + 64) = sub_3887100(v14);
    if ( (unsigned __int8)sub_388AF10(a1, 3, "expected '=' after name") )
      goto LABEL_6;
  }
  if ( (unsigned __int8)sub_388C1F0(a1, (__int64)&v22, &v20, &v23, v24, &v21)
    || (unsigned __int8)sub_388BEC0(a1, v25)
    || (unsigned __int8)sub_388ADB0(a1, v27) )
  {
LABEL_6:
    v16 = 1;
  }
  else if ( (unsigned int)(*(_DWORD *)(a1 + 64) - 91) > 1 )
  {
    v16 = sub_38AA580(a1, v29, v10, v22, v20, v23, a2, a3, a4, a5, v18, v19, a8, a9, v24[0], v21, v25[0], v27[0]);
  }
  else
  {
    v16 = sub_389EEE0(a1, (__int64 *)v29, v10, v22, v23, v24[0], a2, a3, a4, a5, v18, v19, a8, a9, v21, v25[0], v27[0]);
  }
LABEL_7:
  if ( v29[0] != v30 )
    j_j___libc_free_0((unsigned __int64)v29[0]);
  return v16;
}
