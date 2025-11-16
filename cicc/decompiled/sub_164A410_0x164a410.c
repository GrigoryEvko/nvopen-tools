// Function: sub_164A410
// Address: 0x164a410
//
__int64 __fastcall sub_164A410(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  unsigned int v5; // r13d
  unsigned __int8 v7; // al
  __int64 *v8; // r15
  __int64 *v9; // rax
  unsigned __int64 v10; // rdi
  __int64 *v11; // rax
  char v12; // dl
  __int16 v13; // ax
  char v14; // al
  __int64 v15; // r8
  unsigned int v16; // esi
  __int64 v17; // rcx
  unsigned int v18; // r14d
  __int64 v19; // rdi
  unsigned __int64 v20; // rcx
  unsigned int v21; // edx
  __int64 *v22; // r15
  __int64 *v23; // rsi
  __int64 *v24; // rcx
  unsigned __int64 v25; // rax
  __int64 v26; // rdx
  unsigned __int64 v27; // rax
  __int64 v28; // rax
  unsigned __int64 v30; // [rsp+10h] [rbp-A0h] BYREF
  unsigned int v31; // [rsp+18h] [rbp-98h]
  __int64 v32; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v33; // [rsp+28h] [rbp-88h]
  __int64 v34; // [rsp+30h] [rbp-80h] BYREF
  __int64 *v35; // [rsp+38h] [rbp-78h]
  __int64 *v36; // [rsp+40h] [rbp-70h]
  __int64 v37; // [rsp+48h] [rbp-68h]
  int v38; // [rsp+50h] [rbp-60h]
  _QWORD v39[11]; // [rsp+58h] [rbp-58h] BYREF

  v3 = a1;
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 15 )
    return v3;
  v39[0] = a1;
  v5 = *(_DWORD *)(a3 + 8);
  v35 = v39;
  v36 = v39;
  v37 = 0x100000004LL;
  v38 = 0;
  v34 = 1;
LABEL_4:
  v7 = *(_BYTE *)(v3 + 16);
  if ( v7 > 0x17u )
  {
LABEL_5:
    if ( v7 != 56 )
    {
      switch ( v7 )
      {
        case 0x47u:
        case 0x48u:
          goto LABEL_8;
        case 0x4Eu:
          v26 = v3 | 4;
          v27 = v3 & 0xFFFFFFFFFFFFFFF8LL;
          break;
        case 0x1Du:
          v26 = v3 & 0xFFFFFFFFFFFFFFFBLL;
          v27 = v3 & 0xFFFFFFFFFFFFFFF8LL;
          break;
        default:
          goto LABEL_18;
      }
      v32 = v26;
      if ( !v27 )
        goto LABEL_18;
      v28 = sub_14AF150(&v32);
      if ( !v28 )
        goto LABEL_18;
      v3 = v28;
      goto LABEL_11;
    }
    goto LABEL_21;
  }
  while ( 1 )
  {
    if ( v7 != 5 )
    {
      if ( v7 != 1 )
        goto LABEL_18;
      v3 = *(_QWORD *)(v3 - 24);
LABEL_11:
      v9 = v35;
      if ( v36 != v35 )
        goto LABEL_12;
      goto LABEL_39;
    }
    v13 = *(_WORD *)(v3 + 18);
    if ( v13 != 32 )
    {
      if ( v13 != 47 && v13 != 48 )
        goto LABEL_18;
LABEL_8:
      if ( (*(_BYTE *)(v3 + 23) & 0x40) != 0 )
        v8 = *(__int64 **)(v3 - 8);
      else
        v8 = (__int64 *)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF));
      v3 = *v8;
      goto LABEL_11;
    }
LABEL_21:
    if ( (*(_BYTE *)(v3 + 17) & 2) == 0 )
      goto LABEL_18;
    v31 = sub_15A95F0(a2, *(_QWORD *)v3);
    if ( v31 > 0x40 )
      sub_16A4EF0(&v30, 0, 0);
    else
      v30 = 0;
    v14 = sub_1634900(v3, a2, (__int64)&v30);
    v16 = v31;
    if ( !v14 )
      break;
    v17 = v31 - 1;
    v18 = v31 + 1;
    v19 = 1LL << ((unsigned __int8)v31 - 1);
    if ( v31 > 0x40 )
    {
      if ( (*(_QWORD *)(v30 + 8LL * ((unsigned int)v17 >> 6)) & v19) != 0 )
      {
        v16 = v31;
        v17 = (unsigned int)sub_16A5810(&v30);
LABEL_29:
        v21 = v18 - v17;
        goto LABEL_30;
      }
      v16 = v31;
      v21 = v18 - sub_16A57B0(&v30);
    }
    else
    {
      if ( (v19 & v30) != 0 )
      {
        v17 = 64;
        if ( v30 << (64 - (unsigned __int8)v31) != -1 )
        {
          _BitScanReverse64(&v20, ~(v30 << (64 - (unsigned __int8)v31)));
          v17 = (unsigned int)v20 ^ 0x3F;
        }
        goto LABEL_29;
      }
      v21 = 1;
      if ( v30 )
      {
        _BitScanReverse64(&v25, v30);
        v21 = 65 - (v25 ^ 0x3F);
      }
    }
LABEL_30:
    if ( v5 < v21 )
      break;
    sub_16A5D70(&v32, &v30, v5, v17, v15);
    sub_16A7200(a3, &v32);
    if ( v33 > 0x40 && v32 )
      j_j___libc_free_0_0(v32);
    if ( (*(_BYTE *)(v3 + 23) & 0x40) != 0 )
      v22 = *(__int64 **)(v3 - 8);
    else
      v22 = (__int64 *)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF));
    v3 = *v22;
    if ( v31 <= 0x40 || !v30 )
      goto LABEL_11;
    j_j___libc_free_0_0(v30);
    v9 = v35;
    if ( v36 != v35 )
      goto LABEL_12;
LABEL_39:
    v23 = &v9[HIDWORD(v37)];
    if ( v9 != v23 )
    {
      v24 = 0;
      while ( v3 != *v9 )
      {
        if ( *v9 == -2 )
          v24 = v9;
        if ( v23 == ++v9 )
        {
          if ( !v24 )
            goto LABEL_61;
          *v24 = v3;
          --v38;
          ++v34;
          goto LABEL_4;
        }
      }
      return v3;
    }
LABEL_61:
    if ( HIDWORD(v37) < (unsigned int)v37 )
    {
      ++HIDWORD(v37);
      *v23 = v3;
      ++v34;
      goto LABEL_4;
    }
LABEL_12:
    sub_16CCBA0(&v34, v3);
    v10 = (unsigned __int64)v36;
    v11 = v35;
    if ( !v12 )
      goto LABEL_19;
    v7 = *(_BYTE *)(v3 + 16);
    if ( v7 > 0x17u )
      goto LABEL_5;
  }
  if ( v16 > 0x40 && v30 )
    j_j___libc_free_0_0(v30);
LABEL_18:
  v10 = (unsigned __int64)v36;
  v11 = v35;
LABEL_19:
  if ( v11 != (__int64 *)v10 )
    _libc_free(v10);
  return v3;
}
