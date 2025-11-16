// Function: sub_1BA1F80
// Address: 0x1ba1f80
//
__int64 __fastcall sub_1BA1F80(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // rax
  unsigned int v11; // r13d
  __int64 v12; // r15
  __int64 v13; // r14
  unsigned __int64 v14; // r13
  _QWORD *v15; // rax
  double v16; // xmm4_8
  double v17; // xmm5_8
  _QWORD *v18; // r12
  __int64 v19; // rdx
  unsigned __int64 v20; // rax
  __int64 v22; // r14
  __int64 v23; // rax
  __int64 v24; // rax
  _BYTE v25[16]; // [rsp+0h] [rbp-40h] BYREF
  __int16 v26; // [rsp+10h] [rbp-30h]

  v10 = *(_QWORD *)(a1 + 40);
  if ( v10 )
  {
    v11 = *(_DWORD *)(a2 + 12);
    v12 = sub_1BA16F0(a2, **(_QWORD **)(v10 + 40), *(_DWORD *)(a2 + 8));
    if ( *(_BYTE *)(*(_QWORD *)v12 + 8LL) == 16 )
    {
      v22 = *(_QWORD *)(a2 + 176);
      v26 = 257;
      v23 = sub_1643350(*(_QWORD **)(v22 + 24));
      v24 = sub_159C470(v23, v11, 0);
      v12 = sub_156D5F0((__int64 *)v22, v12, v24, (__int64)v25);
    }
  }
  else
  {
    v12 = sub_159C4F0(*(__int64 **)(*(_QWORD *)(a2 + 176) + 24LL));
  }
  v13 = *(_QWORD *)(a2 + 64);
  v14 = sub_157EBA0(v13);
  v15 = sub_1648A60(56, 3u);
  v18 = v15;
  if ( v15 )
    sub_15F83E0((__int64)v15, v13, 0, v12, 0);
  if ( *(v18 - 3) )
  {
    v19 = *(v18 - 2);
    v20 = *(v18 - 1) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v20 = v19;
    if ( v19 )
      *(_QWORD *)(v19 + 16) = *(_QWORD *)(v19 + 16) & 3LL | v20;
  }
  *(v18 - 3) = 0;
  return sub_1AA6530(v14, v18, a3, a4, a5, a6, v16, v17, a9, a10);
}
