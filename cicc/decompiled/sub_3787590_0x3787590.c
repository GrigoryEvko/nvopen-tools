// Function: sub_3787590
// Address: 0x3787590
//
__int64 __fastcall sub_3787590(__int64 *a1, unsigned __int64 a2, __m128i a3)
{
  __int64 v4; // r9
  __int64 v5; // rax
  __int64 v6; // rsi
  _QWORD *v7; // r15
  unsigned int v8; // ebx
  __int64 v9; // r8
  unsigned __int8 *v10; // r15
  __int64 v11; // rdx
  __int64 v12; // rbx
  __int64 v14; // [rsp+8h] [rbp-68h]
  __int128 v15; // [rsp+10h] [rbp-60h] BYREF
  __int128 v16; // [rsp+20h] [rbp-50h] BYREF
  __int64 v17; // [rsp+30h] [rbp-40h] BYREF
  int v18; // [rsp+38h] [rbp-38h]

  HIWORD(v8) = 0;
  *(_QWORD *)&v15 = 0;
  DWORD2(v15) = 0;
  *(_QWORD *)&v16 = 0;
  DWORD2(v16) = 0;
  sub_3780EC0(a1, a2, (__m128i **)&v15, (__m128i **)&v16, 0, a3);
  v5 = *(_QWORD *)(a2 + 48);
  v6 = *(_QWORD *)(a2 + 80);
  v7 = (_QWORD *)a1[1];
  LOWORD(v8) = *(_WORD *)v5;
  v9 = *(_QWORD *)(v5 + 8);
  v17 = v6;
  if ( v6 )
  {
    v14 = v9;
    sub_B96E90((__int64)&v17, v6, 1);
    v9 = v14;
  }
  v18 = *(_DWORD *)(a2 + 72);
  v10 = sub_3406EB0(v7, 0x9Fu, (__int64)&v17, v8, v9, v4, v15, v16);
  v12 = v11;
  if ( v17 )
    sub_B91220((__int64)&v17, v17);
  sub_3760E70((__int64)a1, a2, 0, (unsigned __int64)v10, v12);
  return 0;
}
