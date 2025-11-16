// Function: sub_1D2A660
// Address: 0x1d2a660
//
_QWORD *__fastcall sub_1D2A660(_QWORD *a1, int a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 v8; // bl
  _QWORD *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r9
  __int64 v13; // r8
  _QWORD *v14; // r12
  unsigned __int8 v16; // r14
  __int64 v17; // rbx
  __int128 v18; // rdi
  __int64 v19; // rax
  unsigned __int8 *v20; // rsi
  __int64 v21; // rdi
  __int64 (*v22)(); // r8
  char v23; // al
  __int64 *v24; // rdx
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v28; // [rsp+8h] [rbp-D8h]
  __int64 *v29; // [rsp+10h] [rbp-D0h] BYREF
  unsigned __int8 *v30; // [rsp+18h] [rbp-C8h] BYREF
  unsigned __int64 v31[2]; // [rsp+20h] [rbp-C0h] BYREF
  _BYTE v32[176]; // [rsp+30h] [rbp-B0h] BYREF

  v8 = a3;
  v31[0] = (unsigned __int64)v32;
  v31[1] = 0x2000000000LL;
  v28 = sub_1D29190((__int64)a1, a3, a4, a4, a5, a6);
  sub_16BD430((__int64)v31, 8);
  sub_16BD4C0((__int64)v31, v28);
  sub_16BD430((__int64)v31, a2);
  v29 = 0;
  v9 = sub_1D17910((__int64)a1, (__int64)v31, (__int64 *)&v29);
  v13 = a4;
  if ( v9 )
  {
    v14 = v9;
  }
  else
  {
    v16 = v8;
    v17 = a1[26];
    if ( v17 )
    {
      a1[26] = *(_QWORD *)v17;
    }
    else
    {
      v26 = sub_145CBF0(a1 + 27, 112, 8);
      v13 = a4;
      v17 = v26;
    }
    *((_QWORD *)&v18 + 1) = v13;
    *(_QWORD *)&v18 = v16;
    v19 = sub_1D274F0(v18, v10, v11, v13, v12);
    v30 = 0;
    *(_QWORD *)v17 = 0;
    v20 = v30;
    *(_QWORD *)(v17 + 40) = v19;
    *(_QWORD *)(v17 + 8) = 0;
    *(_QWORD *)(v17 + 16) = 0;
    *(_WORD *)(v17 + 24) = 8;
    *(_DWORD *)(v17 + 28) = -1;
    *(_QWORD *)(v17 + 32) = 0;
    *(_QWORD *)(v17 + 48) = 0;
    *(_QWORD *)(v17 + 56) = 0x100000000LL;
    *(_DWORD *)(v17 + 64) = 0;
    *(_QWORD *)(v17 + 72) = v20;
    if ( v20 )
      sub_1623210((__int64)&v30, v20, v17 + 72);
    *(_WORD *)(v17 + 80) &= 0xF000u;
    *(_WORD *)(v17 + 26) = 0;
    *(_DWORD *)(v17 + 84) = a2;
    v21 = a1[2];
    v22 = *(__int64 (**)())(*(_QWORD *)v21 + 984LL);
    v23 = 0;
    if ( v22 != sub_1D12E30 )
      v23 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, _QWORD))v22)(v21, v17, a1[9], a1[8]);
    v24 = v29;
    *(_BYTE *)(v17 + 26) = *(_BYTE *)(v17 + 26) & 0xFB | (4 * (v23 & 1));
    sub_16BDA20(a1 + 40, (__int64 *)v17, v24);
    v25 = (__int64)a1;
    v14 = (_QWORD *)v17;
    sub_1D172A0(v25, v17);
  }
  if ( (_BYTE *)v31[0] != v32 )
    _libc_free(v31[0]);
  return v14;
}
