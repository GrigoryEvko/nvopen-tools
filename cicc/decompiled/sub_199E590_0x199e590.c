// Function: sub_199E590
// Address: 0x199e590
//
__int64 __fastcall sub_199E590(__int64 a1, __int64 a2, __m128i a3, __m128i a4)
{
  _QWORD *v6; // r13
  __int16 v7; // ax
  __int64 v8; // r12
  __int64 v10; // r12
  void *v11; // r9
  signed __int64 v12; // r12
  __int64 v13; // r8
  __int64 *v14; // rax
  __int64 *v15; // rdi
  _BYTE *v16; // rsi
  _BYTE *v17; // rdx
  void *src; // [rsp+0h] [rbp-90h]
  int v19; // [rsp+8h] [rbp-88h]
  __int64 *v20; // [rsp+10h] [rbp-80h] BYREF
  __int64 v21; // [rsp+18h] [rbp-78h]
  _BYTE v22[112]; // [rsp+20h] [rbp-70h] BYREF

  v6 = *(_QWORD **)a1;
  v7 = *(_WORD *)(*(_QWORD *)a1 + 24LL);
  if ( v7 == 10 )
  {
    v8 = *(v6 - 1);
    if ( *(_BYTE *)(v8 + 16) > 3u )
      return 0;
    *(_QWORD *)a1 = sub_145CF80(a2, *(_QWORD *)v8, 0, 0);
    return v8;
  }
  if ( v7 != 4 )
  {
    v8 = 0;
    if ( v7 != 7 )
      return v8;
    v16 = (_BYTE *)v6[4];
    v17 = &v16[8 * v6[5]];
    v20 = (__int64 *)v22;
    v21 = 0x800000000LL;
    sub_145C5B0((__int64)&v20, v16, v17);
    v8 = sub_199E590(v20, a2);
    if ( v8 )
      *(_QWORD *)a1 = sub_14785F0(a2, &v20, v6[6], 0);
    goto LABEL_10;
  }
  v10 = v6[5];
  v11 = (void *)v6[4];
  v20 = (__int64 *)v22;
  v12 = 8 * v10;
  v21 = 0x800000000LL;
  v13 = v12 >> 3;
  if ( (unsigned __int64)v12 > 0x40 )
  {
    src = v11;
    sub_16CD150((__int64)&v20, v22, v12 >> 3, 8, v13, (int)v11);
    v13 = v12 >> 3;
    v11 = src;
    v15 = &v20[(unsigned int)v21];
  }
  else
  {
    v14 = (__int64 *)v22;
    if ( !v12 )
      goto LABEL_8;
    v15 = (__int64 *)v22;
  }
  v19 = v13;
  memcpy(v15, v11, v12);
  v14 = v20;
  LODWORD(v12) = v21;
  LODWORD(v13) = v19;
LABEL_8:
  LODWORD(v21) = v13 + v12;
  v8 = sub_199E590(&v14[(unsigned int)(v13 + v12) - 1], a2);
  if ( v8 )
    *(_QWORD *)a1 = sub_147DD40(a2, (__int64 *)&v20, 0, 0, a3, a4);
LABEL_10:
  if ( v20 != (__int64 *)v22 )
    _libc_free((unsigned __int64)v20);
  return v8;
}
