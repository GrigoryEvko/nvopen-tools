// Function: sub_37A8AE0
// Address: 0x37a8ae0
//
__int64 __fastcall sub_37A8AE0(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  _QWORD *v4; // r12
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r9
  __int64 v8; // rsi
  __int128 *v9; // rcx
  __int64 v10; // r10
  __int64 v11; // rax
  __int64 v12; // r11
  unsigned int v13; // r15d
  __int64 v14; // r8
  unsigned int v15; // esi
  __int64 v16; // r12
  __int128 v18; // [rsp-20h] [rbp-80h]
  __int64 v19; // [rsp+0h] [rbp-60h]
  __int64 v20; // [rsp+8h] [rbp-58h]
  __int64 v21; // [rsp+10h] [rbp-50h]
  __int128 *v22; // [rsp+18h] [rbp-48h]
  __int64 v23; // [rsp+20h] [rbp-40h] BYREF
  int v24; // [rsp+28h] [rbp-38h]

  HIWORD(v13) = 0;
  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD **)(a1 + 8);
  v5 = sub_379AB60(a1, *(_QWORD *)(v3 + 40), *(_QWORD *)(v3 + 48));
  v8 = *(_QWORD *)(a2 + 80);
  v9 = *(__int128 **)(a2 + 40);
  v10 = v5;
  v11 = *(_QWORD *)(a2 + 48);
  v12 = v6;
  LOWORD(v13) = *(_WORD *)v11;
  v14 = *(_QWORD *)(v11 + 8);
  v23 = v8;
  if ( v8 )
  {
    v20 = v6;
    v19 = v10;
    v21 = v14;
    v22 = v9;
    sub_B96E90((__int64)&v23, v8, 1);
    v10 = v19;
    v12 = v20;
    v14 = v21;
    v9 = v22;
  }
  v15 = *(_DWORD *)(a2 + 24);
  v24 = *(_DWORD *)(a2 + 72);
  *((_QWORD *)&v18 + 1) = v12;
  *(_QWORD *)&v18 = v10;
  v16 = sub_340F900(v4, v15, (__int64)&v23, v13, v14, v7, *v9, v18, *(_OWORD *)(v3 + 80));
  if ( v23 )
    sub_B91220((__int64)&v23, v23);
  return v16;
}
