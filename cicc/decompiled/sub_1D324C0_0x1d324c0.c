// Function: sub_1D324C0
// Address: 0x1d324c0
//
__int64 __fastcall sub_1D324C0(
        __int64 *a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        double a6,
        double a7,
        double a8)
{
  __int64 v11; // rax
  __int64 v12; // r9
  char v13; // r15
  __int64 v14; // rdi
  __int64 v15; // rax
  unsigned int v16; // eax
  const void **v17; // rdx
  const void **v18; // r9
  unsigned int v19; // ebx
  __int64 result; // rax
  __int64 v21; // rsi
  char v22; // al
  __int64 v24; // [rsp+18h] [rbp-58h]
  const void **v25; // [rsp+18h] [rbp-58h]
  __int64 v26; // [rsp+18h] [rbp-58h]
  const void **v27; // [rsp+18h] [rbp-58h]
  char v28[8]; // [rsp+20h] [rbp-50h] BYREF
  const void **v29; // [rsp+28h] [rbp-48h]
  __int64 v30; // [rsp+30h] [rbp-40h] BYREF
  int v31; // [rsp+38h] [rbp-38h]

  v11 = *(_QWORD *)(a4 + 40) + 16LL * (unsigned int)a5;
  v12 = a1[2];
  v13 = *(_BYTE *)v11;
  v14 = a1[4];
  v24 = v12;
  v29 = *(const void ***)(v11 + 8);
  v28[0] = v13;
  v15 = sub_1E0A0C0(v14);
  v16 = sub_1F40B60(v24, a2, a3, v15, 1);
  v18 = v17;
  v19 = v16;
  if ( v13 == (_BYTE)v16 )
  {
    if ( v29 == v17 || v13 )
      return a4;
LABEL_12:
    v27 = v17;
    v22 = sub_1F58D20(v28);
    v18 = v27;
    if ( !v22 )
      goto LABEL_8;
    return a4;
  }
  if ( !v13 )
    goto LABEL_12;
  if ( (unsigned __int8)(v13 - 14) <= 0x5Fu )
    return a4;
LABEL_8:
  v21 = *(_QWORD *)(a4 + 72);
  v30 = v21;
  if ( v21 )
  {
    v25 = v18;
    sub_1623A60((__int64)&v30, v21, 2);
    v18 = v25;
  }
  v31 = *(_DWORD *)(a4 + 64);
  result = sub_1D323C0(a1, a4, a5, (__int64)&v30, v19, v18, a6, a7, a8);
  if ( v30 )
  {
    v26 = result;
    sub_161E7C0((__int64)&v30, v30);
    return v26;
  }
  return result;
}
