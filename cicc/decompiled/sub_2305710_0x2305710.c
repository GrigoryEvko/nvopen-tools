// Function: sub_2305710
// Address: 0x2305710
//
__int64 *__fastcall sub_2305710(__int64 *a1, __int64 a2)
{
  __int64 v3; // r13
  unsigned int v4; // ebx
  __int64 v5; // r15
  int v6; // r14d
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rsi
  __int64 v10; // rdi
  __int16 v12; // [rsp+6h] [rbp-6Ah]
  char v13; // [rsp+8h] [rbp-68h]
  __int64 v14; // [rsp+8h] [rbp-68h]
  __int64 v15; // [rsp+10h] [rbp-60h] BYREF
  __int64 v16; // [rsp+18h] [rbp-58h]
  __int64 v17; // [rsp+20h] [rbp-50h]
  unsigned int v18; // [rsp+28h] [rbp-48h]
  int v19; // [rsp+30h] [rbp-40h]
  __int16 v20; // [rsp+34h] [rbp-3Ch]
  char v21; // [rsp+36h] [rbp-3Ah]

  sub_2FCE3D0(&v15, a2 + 8);
  v3 = v16;
  v4 = v18;
  v5 = v17;
  v16 = 0;
  v13 = v21;
  ++v15;
  v6 = v19;
  v17 = 0;
  v18 = 0;
  v12 = v20;
  v7 = sub_22077B0(0x30u);
  if ( v7 )
  {
    *(_DWORD *)(v7 + 32) = v4;
    v8 = 0;
    *(_QWORD *)(v7 + 8) = 1;
    *(_WORD *)(v7 + 44) = v12;
    *(_QWORD *)(v7 + 24) = v5;
    *(_QWORD *)v7 = &unk_4A0ADB8;
    *(_DWORD *)(v7 + 40) = v6;
    *(_BYTE *)(v7 + 46) = v13;
    *(_QWORD *)(v7 + 16) = v3;
    v3 = 0;
  }
  else
  {
    v8 = 16LL * v4;
  }
  v14 = v7;
  sub_C7D6A0(v3, v8, 8);
  v9 = v18;
  v10 = v16;
  *a1 = v14;
  sub_C7D6A0(v10, 16 * v9, 8);
  return a1;
}
