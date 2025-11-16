// Function: sub_1904E90
// Address: 0x1904e90
//
__int64 __fastcall sub_1904E90(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        double a7,
        double a8,
        double a9)
{
  __int64 *v12; // rdi
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v18; // rax
  __int64 v19; // rdx
  char v20; // al
  int v21; // r8d
  __int64 v22; // rdi
  __int64 *v23; // r14
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 v26; // rsi
  __int64 v27; // rsi
  unsigned __int8 *v28; // rsi
  int v30; // [rsp+1Ch] [rbp-54h]
  _QWORD v31[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v32; // [rsp+30h] [rbp-40h]

  v12 = (__int64 *)a2;
  v14 = a4;
  if ( *(_BYTE *)(a3 + 16) > 0x10u
    || *(_BYTE *)(a4 + 16) > 0x10u
    || (v15 = sub_15A2A30(v12, (__int64 *)a3, a4, 0, 0, a7, a8, a9), v14 = a4, (v16 = v15) == 0) )
  {
    v32 = 257;
    v18 = sub_15FB440((int)v12, (__int64 *)a3, v14, (__int64)v31, 0);
    v19 = *(_QWORD *)v18;
    v16 = v18;
    v20 = *(_BYTE *)(*(_QWORD *)v18 + 8LL);
    if ( v20 == 16 )
      v20 = *(_BYTE *)(**(_QWORD **)(v19 + 16) + 8LL);
    if ( (unsigned __int8)(v20 - 1) <= 5u || *(_BYTE *)(v16 + 16) == 76 )
    {
      v21 = *(_DWORD *)(a1 + 40);
      if ( a6 || (a6 = *(_QWORD *)(a1 + 32)) != 0 )
      {
        v30 = *(_DWORD *)(a1 + 40);
        sub_1625C10(v16, 3, a6);
        v21 = v30;
      }
      sub_15F2440(v16, v21);
    }
    v22 = *(_QWORD *)(a1 + 8);
    if ( v22 )
    {
      v23 = *(__int64 **)(a1 + 16);
      sub_157E9D0(v22 + 40, v16);
      v24 = *(_QWORD *)(v16 + 24);
      v25 = *v23;
      *(_QWORD *)(v16 + 32) = v23;
      v25 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v16 + 24) = v25 | v24 & 7;
      *(_QWORD *)(v25 + 8) = v16 + 24;
      *v23 = *v23 & 7 | (v16 + 24);
    }
    sub_164B780(v16, a5);
    v26 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
    {
      v31[0] = *(_QWORD *)a1;
      sub_1623A60((__int64)v31, v26, 2);
      v27 = *(_QWORD *)(v16 + 48);
      if ( v27 )
        sub_161E7C0(v16 + 48, v27);
      v28 = (unsigned __int8 *)v31[0];
      *(_QWORD *)(v16 + 48) = v31[0];
      if ( v28 )
        sub_1623210((__int64)v31, v28, v16 + 48);
    }
  }
  return v16;
}
