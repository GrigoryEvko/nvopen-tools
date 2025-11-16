// Function: sub_1728E00
// Address: 0x1728e00
//
__int64 __fastcall sub_1728E00(_QWORD *a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v4; // r14
  int v8; // edi
  bool v9; // al
  unsigned int v10; // edi
  __int64 v11; // rbx
  __int64 *v12; // r14
  __int64 *v13; // r10
  unsigned int v14; // eax
  __int64 v15; // r14
  int v16; // r11d
  __int64 v17; // r10
  __int16 v18; // r11
  __int64 v19; // rdx
  __int64 v20; // r12
  __int64 v21; // rax
  unsigned int v23; // edx
  __int16 v24; // ax
  int v25; // [rsp+4h] [rbp-7Ch]
  int v26; // [rsp+8h] [rbp-78h]
  __int64 *v27; // [rsp+8h] [rbp-78h]
  int v28; // [rsp+8h] [rbp-78h]
  __int64 v29; // [rsp+10h] [rbp-70h] BYREF
  __int16 v30; // [rsp+20h] [rbp-60h]
  __int64 v31; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v32; // [rsp+38h] [rbp-48h]
  __int64 v33; // [rsp+40h] [rbp-40h]
  unsigned int v34; // [rsp+48h] [rbp-38h]

  v4 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v4 + 16) != 13 )
    return 0;
  v8 = *(_WORD *)(a2 + 18) & 0x7FFF;
  if ( a4 )
  {
    v8 = sub_15FF0F0(v8);
    if ( v8 != 38 )
      goto LABEL_4;
LABEL_32:
    v23 = *(_DWORD *)(v4 + 32);
    if ( v23 <= 0x40 )
    {
      v9 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v23) == *(_QWORD *)(v4 + 24);
    }
    else
    {
      v28 = *(_DWORD *)(v4 + 32);
      v9 = v28 == (unsigned int)sub_16A58F0(v4 + 24);
    }
    goto LABEL_7;
  }
  if ( v8 == 38 )
    goto LABEL_32;
LABEL_4:
  if ( v8 != 39 )
    return 0;
  if ( *(_DWORD *)(v4 + 32) <= 0x40u )
  {
    v9 = *(_QWORD *)(v4 + 24) == 0;
  }
  else
  {
    v26 = *(_DWORD *)(v4 + 32);
    v9 = v26 == (unsigned int)sub_16A57B0(v4 + 24);
  }
LABEL_7:
  if ( !v9 )
    return 0;
  v10 = *(_WORD *)(a3 + 18) & 0x7FFF;
  if ( a4 )
    v10 = sub_15FF0F0(v10);
  v11 = *(_QWORD *)(a2 - 48);
  v12 = *(__int64 **)(a3 - 48);
  v13 = *(__int64 **)(a3 - 24);
  if ( v12 != (__int64 *)v11 )
  {
    if ( (__int64 *)v11 == v13 )
    {
      v14 = sub_15FF5D0(v10);
      v13 = v12;
      v10 = v14;
      goto LABEL_13;
    }
    return 0;
  }
LABEL_13:
  if ( v10 == 40 )
  {
    v16 = 36;
  }
  else
  {
    v15 = 0;
    v16 = 37;
    if ( v10 != 41 )
      return v15;
  }
  v25 = v16;
  v27 = v13;
  sub_14C2530((__int64)&v31, v13, a1[333], 0, a1[330], a3, a1[332], 0);
  v17 = (__int64)v27;
  v18 = v25;
  if ( v32 > 0x40 )
    v19 = *(_QWORD *)(v31 + 8LL * ((v32 - 1) >> 6));
  else
    v19 = v31;
  v15 = 0;
  if ( (v19 & (1LL << ((unsigned __int8)v32 - 1))) != 0 )
  {
    if ( a4 )
    {
      v24 = sub_15FF0F0(v25);
      v17 = (__int64)v27;
      v18 = v24;
    }
    v20 = a1[1];
    v30 = 257;
    if ( *(_BYTE *)(v11 + 16) > 0x10u || *(_BYTE *)(v17 + 16) > 0x10u )
    {
      v15 = (__int64)sub_1727440(v20, v18, v11, v17, &v29);
    }
    else
    {
      v15 = sub_15A37B0(v18, (_QWORD *)v11, (_QWORD *)v17, 0);
      v21 = sub_14DBA30(v15, *(_QWORD *)(v20 + 96), 0);
      if ( v21 )
        v15 = v21;
    }
  }
  if ( v34 > 0x40 && v33 )
    j_j___libc_free_0_0(v33);
  if ( v32 > 0x40 && v31 )
    j_j___libc_free_0_0(v31);
  return v15;
}
