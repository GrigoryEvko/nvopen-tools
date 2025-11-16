// Function: sub_3862E50
// Address: 0x3862e50
//
__int64 __fastcall sub_3862E50(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        char a4,
        int a5,
        int a6,
        __m128i a7,
        __m128i a8,
        __int64 a9,
        __int64 a10)
{
  __int64 *v10; // rax
  _QWORD *v11; // r12
  __int64 v12; // r13
  bool v13; // al
  __int64 v14; // r11
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 *v17; // r14
  __int64 v18; // rax
  __int64 v19; // r11
  __int64 v20; // rdx
  unsigned int v21; // esi
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 *v25; // rax
  __int64 v26; // r14
  unsigned int v27; // eax
  __int64 v28; // r12
  __int64 result; // rax
  __int64 *v30; // rax
  __int64 v31; // rax
  __int64 v32; // [rsp+0h] [rbp-70h]
  unsigned int v37; // [rsp+18h] [rbp-58h]
  __int64 v38; // [rsp+18h] [rbp-58h]
  __int64 v39[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v40[8]; // [rsp+30h] [rbp-40h] BYREF

  v10 = sub_385D990(a10, a9, a3, 0, a7, a8);
  v11 = *(_QWORD **)(a10 + 112);
  v12 = (__int64)v10;
  v13 = sub_146CEE0((__int64)v11, (__int64)v10, a2);
  v14 = a3;
  if ( v13 )
  {
    v26 = v12;
    v16 = v12;
    goto LABEL_11;
  }
  if ( *(_WORD *)(v12 + 24) != 7 )
  {
    sub_1495DC0(a10, a7, a8);
    BUG();
  }
  v15 = sub_1495DC0(a10, a7, a8);
  v16 = **(_QWORD **)(v12 + 32);
  v17 = sub_1487810(v12, v15, v11, a7, a8);
  v18 = sub_13A5BC0((_QWORD *)v12, (__int64)v11);
  v19 = a3;
  if ( *(_WORD *)(v18 + 24) )
  {
    v16 = sub_1481BD0(v11, v16, (__int64)v17, a7, a8);
    v31 = sub_14819D0(v11, **(_QWORD **)(v12 + 32), (__int64)v17, a7, a8);
    v19 = a3;
    v17 = (__int64 *)v31;
  }
  else
  {
    v20 = *(_QWORD *)(v18 + 32);
    v21 = *(_DWORD *)(v20 + 32);
    v22 = *(_QWORD *)(v20 + 24);
    v23 = 1LL << ((unsigned __int8)v21 - 1);
    if ( v21 <= 0x40 )
    {
      if ( (v22 & v23) == 0 )
        goto LABEL_6;
      goto LABEL_18;
    }
    if ( (*(_QWORD *)(v22 + 8LL * ((v21 - 1) >> 6)) & v23) != 0 )
    {
LABEL_18:
      v30 = (__int64 *)v16;
      v16 = (__int64)v17;
      v17 = v30;
    }
  }
LABEL_6:
  v32 = v19;
  v37 = sub_16431D0(**(_QWORD **)(*(_QWORD *)v19 + 16LL));
  v24 = sub_1456040((__int64)v17);
  v40[1] = sub_145CF80((__int64)v11, v24, v37 >> 3, 0);
  v39[0] = (__int64)v40;
  v40[0] = v17;
  v39[1] = 0x200000002LL;
  v25 = sub_147DD40((__int64)v11, v39, 0, 0, a7, a8);
  v14 = v32;
  v26 = (__int64)v25;
  if ( (_QWORD *)v39[0] != v40 )
  {
    _libc_free(v39[0]);
    v14 = v32;
    v27 = *(_DWORD *)(a1 + 16);
    if ( v27 >= *(_DWORD *)(a1 + 20) )
      goto LABEL_8;
    goto LABEL_12;
  }
LABEL_11:
  v27 = *(_DWORD *)(a1 + 16);
  if ( v27 >= *(_DWORD *)(a1 + 20) )
  {
LABEL_8:
    v38 = v14;
    sub_3862C60(a1 + 8, 0);
    v27 = *(_DWORD *)(a1 + 16);
    v14 = v38;
    v28 = *(_QWORD *)(a1 + 8) + ((unsigned __int64)v27 << 6);
    if ( !v28 )
      goto LABEL_9;
    goto LABEL_13;
  }
LABEL_12:
  v28 = *(_QWORD *)(a1 + 8) + ((unsigned __int64)v27 << 6);
  if ( !v28 )
    goto LABEL_9;
LABEL_13:
  *(_QWORD *)v28 = 6;
  *(_QWORD *)(v28 + 8) = 0;
  if ( v14 )
  {
    *(_QWORD *)(v28 + 16) = v14;
    if ( v14 != -16 && v14 != -8 )
      sub_164C220(v28);
  }
  else
  {
    *(_QWORD *)(v28 + 16) = 0;
  }
  *(_QWORD *)(v28 + 24) = v16;
  *(_QWORD *)(v28 + 32) = v26;
  *(_BYTE *)(v28 + 40) = a4;
  *(_QWORD *)(v28 + 56) = v12;
  *(_DWORD *)(v28 + 44) = a5;
  *(_DWORD *)(v28 + 48) = a6;
  v27 = *(_DWORD *)(a1 + 16);
LABEL_9:
  result = v27 + 1;
  *(_DWORD *)(a1 + 16) = result;
  return result;
}
