// Function: sub_6F1D40
// Address: 0x6f1d40
//
__int64 __fastcall sub_6F1D40(__int64 a1, __int64 a2, int a3, __int64 a4, __m128i *a5)
{
  __int64 *v6; // r14
  __int64 v7; // r13
  __int64 v8; // r15
  __int64 v9; // rax
  char v10; // bl
  bool v11; // bl
  _QWORD *v12; // rax
  __int64 v13; // rdi
  unsigned int v14; // r13d
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  int v21; // [rsp+2Ch] [rbp-44h] BYREF
  _QWORD v22[8]; // [rsp+30h] [rbp-40h] BYREF

  v6 = *(__int64 **)(a2 + 56);
  v21 = 0;
  v7 = *v6;
  v8 = **(_QWORD **)(*(_QWORD *)(*v6 + 88) + 32LL);
  sub_865900(*v6);
  v9 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v10 = *(_BYTE *)(v9 + 12);
  *(_BYTE *)(v9 + 12) = v10 | 0x20;
  v11 = (v10 & 0x20) != 0;
  v12 = (_QWORD *)sub_725090(0);
  v12[4] = a1;
  *v12 = *(_QWORD *)(a2 + 64);
  v13 = v7;
  v14 = 0;
  v15 = sub_8A6360(v13, (_DWORD)v12, v8, 0, a3, (int)a2 + 28, 0, (__int64)&v21, a4);
  if ( !v21 )
  {
    v13 = v6[24];
    if ( !a5 )
    {
      v22[0] = 0;
      a5 = (__m128i *)v22;
      v22[1] = 0;
    }
    v14 = sub_6F1C10(v13, v15, v8, a5, 0, 0, 0, 0);
  }
  v16 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v17 = *(_BYTE *)(v16 + 12) & 0xDF;
  *(_BYTE *)(v16 + 12) = *(_BYTE *)(v16 + 12) & 0xDF | (32 * v11);
  sub_864110(v13, qword_4F04C68, v17);
  return v14;
}
