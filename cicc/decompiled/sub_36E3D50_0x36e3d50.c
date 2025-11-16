// Function: sub_36E3D50
// Address: 0x36e3d50
//
__int64 __fastcall sub_36E3D50(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // rcx
  __int64 v6; // rdx
  _QWORD *v7; // rax
  unsigned int v8; // r13d
  _QWORD *v10; // rdi
  __int64 v11; // r8
  __m128i v12; // xmm1
  unsigned __int64 v13; // rcx
  __int64 v14; // r13
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rdi
  __int64 v19; // [rsp+0h] [rbp-50h] BYREF
  int v20; // [rsp+8h] [rbp-48h]
  _OWORD v21[4]; // [rsp+10h] [rbp-40h] BYREF

  v4 = *(_QWORD *)(a2 + 80);
  v19 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v19, v4, 1);
  v5 = *(_QWORD *)(a2 + 40);
  v20 = *(_DWORD *)(a2 + 72);
  v6 = *(_QWORD *)(*(_QWORD *)(v5 + 40) + 96LL);
  v7 = *(_QWORD **)(v6 + 24);
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
    v7 = (_QWORD *)*v7;
  v8 = 0;
  if ( (_DWORD)v7 == 3 )
  {
    v10 = *(_QWORD **)(a1 + 64);
    v11 = *(unsigned int *)(a2 + 68);
    v21[0] = _mm_loadu_si128((const __m128i *)(v5 + 120));
    v12 = _mm_loadu_si128((const __m128i *)v5);
    v13 = *(_QWORD *)(a2 + 48);
    v21[1] = v12;
    v14 = sub_33E66D0(v10, 5594, (__int64)&v19, v13, v11, (__int64)&v19, (unsigned __int64 *)v21, 2);
    sub_34158F0(*(_QWORD *)(a1 + 64), a2, v14, v15, v16, v17);
    v18 = v14;
    v8 = 1;
    sub_3421DB0(v18);
    sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  }
  if ( v19 )
    sub_B91220((__int64)&v19, v19);
  return v8;
}
