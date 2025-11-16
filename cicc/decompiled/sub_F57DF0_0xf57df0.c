// Function: sub_F57DF0
// Address: 0xf57df0
//
__int64 __fastcall sub_F57DF0(__int64 a1, char *a2, __int64 a3)
{
  char v6; // dl
  unsigned __int8 v7; // cl
  __int64 result; // rax
  unsigned int v9; // ebx
  unsigned __int64 v10; // r13
  __int64 v11; // rdx
  __int64 v12; // rdx
  int v13; // esi
  char v14; // cl
  __int64 v15; // rdx
  unsigned __int64 v16; // rsi
  unsigned int v17; // r13d
  unsigned __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // rdi
  int v21; // eax
  __int64 v22; // rax
  __int64 *v23; // rdi
  int v24; // edx
  char v25; // cl
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  __int64 *v28; // rsi
  __int64 *v29; // rax
  _QWORD *v30; // rdx
  __int64 v31; // rax
  __int64 *v32; // rdi
  unsigned __int64 v33; // rdx
  __int64 v34; // rdi
  int v35; // edx
  unsigned __int64 v36; // rax
  __int64 v37; // [rsp+8h] [rbp-58h]
  __int64 *v38; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v39; // [rsp+18h] [rbp-48h]
  __int64 v40; // [rsp+20h] [rbp-40h] BYREF
  __int64 v41; // [rsp+28h] [rbp-38h]
  __int64 v42; // [rsp+30h] [rbp-30h]

  v6 = *a2;
  if ( *a2 == 17 )
  {
    v17 = *((_DWORD *)a2 + 8);
    v18 = *((_QWORD *)a2 + 3);
    v19 = 1LL << ((unsigned __int8)v17 - 1);
    if ( v17 > 0x40 )
    {
      v20 = (__int64)(a2 + 24);
      if ( (*(_QWORD *)(v18 + 8LL * ((v17 - 1) >> 6)) & v19) != 0 )
        v21 = sub_C44500(v20);
      else
        v21 = sub_C444A0(v20);
      if ( v17 + 1 - v21 > 0x40 )
        return 0;
      v22 = *(_QWORD *)v18;
      goto LABEL_28;
    }
    v22 = v18 & v19;
    if ( v22 )
    {
      if ( !v17 )
      {
        v22 = 0;
        goto LABEL_28;
      }
      v24 = 64;
      v25 = 64 - v17;
      v26 = v18 << (64 - (unsigned __int8)v17);
      if ( v26 != -1 )
      {
        _BitScanReverse64(&v27, ~(v18 << (64 - (unsigned __int8)v17)));
        v24 = v27 ^ 0x3F;
      }
      if ( v17 + 1 - v24 > 0x40 )
        return 0;
    }
    else
    {
      if ( v18 )
      {
        _BitScanReverse64(&v33, v18);
        if ( (unsigned int)v33 == 0x3F )
          return 0;
      }
      if ( !v17 )
        goto LABEL_28;
      v25 = 64 - v17;
      v26 = v18 << (64 - (unsigned __int8)v17);
    }
    v22 = v26 >> v25;
LABEL_28:
    v40 = 16;
    v23 = *(__int64 **)(a1 + 8);
    v41 = v22;
    goto LABEL_29;
  }
  v7 = *(_BYTE *)(a3 + 8);
  if ( v6 != 18 )
  {
    result = 0;
    if ( v7 != 14 )
      return result;
    if ( v6 != 20 )
    {
      if ( v6 != 5 )
        return result;
      if ( *((_WORD *)a2 + 1) != 48 )
        return result;
      result = *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      if ( !result )
        return result;
      if ( *(_BYTE *)result != 17 )
        return 0;
      v9 = *(_DWORD *)(result + 32);
      v10 = *(_QWORD *)(result + 24);
      v11 = 1LL << ((unsigned __int8)v9 - 1);
      if ( v9 > 0x40 )
      {
        v34 = result + 24;
        if ( (*(_QWORD *)(v10 + 8LL * ((v9 - 1) >> 6)) & v11) != 0 )
          v35 = sub_C44500(v34);
        else
          v35 = sub_C444A0(v34);
        if ( v9 + 1 - v35 > 0x40 )
          return 0;
        v12 = *(_QWORD *)v10;
        goto LABEL_54;
      }
      v12 = v10 & v11;
      if ( v12 )
      {
        if ( !v9 )
        {
          v12 = 0;
          goto LABEL_54;
        }
        v13 = 64;
        v14 = 64 - v9;
        v15 = v10 << (64 - (unsigned __int8)v9);
        if ( v15 != -1 )
        {
          _BitScanReverse64(&v16, ~(v10 << (64 - (unsigned __int8)v9)));
          v13 = v16 ^ 0x3F;
        }
        if ( v9 + 1 - v13 > 0x40 )
          return 0;
      }
      else
      {
        if ( v10 )
        {
          _BitScanReverse64(&v36, v10);
          if ( (unsigned int)v36 == 0x3F )
            return 0;
        }
        if ( !v9 )
          goto LABEL_54;
        v14 = 64 - v9;
        v15 = v10 << (64 - (unsigned __int8)v9);
      }
      v12 = v15 >> v14;
LABEL_54:
      v40 = 16;
      v23 = *(__int64 **)(a1 + 8);
      v41 = v12;
      goto LABEL_29;
    }
    v40 = 16;
    v23 = *(__int64 **)(a1 + 8);
    v41 = 0;
LABEL_29:
    v42 = 159;
    return sub_B0D000(v23, &v40, 3, 0, 1);
  }
  if ( v7 > 3u && v7 != 5 )
  {
    result = 0;
    if ( (v7 & 0xFD) != 4 )
      return result;
  }
  if ( (unsigned int)sub_BCB060(a3) > 0x40 )
    return 0;
  v28 = (__int64 *)(a2 + 24);
  if ( *((void **)a2 + 3) == sub_C33340() )
    sub_C3E660((__int64)&v38, (__int64)v28);
  else
    sub_C3A850((__int64)&v38, v28);
  v29 = v38;
  if ( v39 <= 0x40 )
  {
    v30 = v38;
    v29 = (__int64 *)&v38;
    if ( !v38 )
      goto LABEL_40;
  }
  else
  {
    v30 = (_QWORD *)*v38;
    if ( !*v38 )
    {
LABEL_40:
      v31 = *v29;
      v32 = *(__int64 **)(a1 + 8);
      v40 = 16;
      v41 = v31;
      goto LABEL_41;
    }
  }
  v40 = 16;
  v32 = *(__int64 **)(a1 + 8);
  v41 = (__int64)v30;
LABEL_41:
  v42 = 159;
  result = sub_B0D000(v32, &v40, 3, 0, 1);
  if ( v39 > 0x40 )
  {
    if ( v38 )
    {
      v37 = result;
      j_j___libc_free_0_0(v38);
      return v37;
    }
  }
  return result;
}
