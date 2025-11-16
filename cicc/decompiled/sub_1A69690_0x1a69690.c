// Function: sub_1A69690
// Address: 0x1a69690
//
unsigned __int64 __fastcall sub_1A69690(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 v8; // rax
  unsigned __int64 result; // rax
  __int64 v10; // rdx
  __int64 *v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rdx
  unsigned __int64 v14; // rdx
  void *v15; // rcx
  void *v16; // rcx
  __int64 v17; // rcx
  unsigned __int64 *v18; // rsi
  __int64 v19; // rbx
  unsigned int v20; // eax
  __int64 v21; // r10
  __int64 *v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // [rsp+10h] [rbp-60h]
  __int64 v26; // [rsp+18h] [rbp-58h]
  unsigned __int64 v27; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v28; // [rsp+28h] [rbp-48h]
  unsigned __int64 v29; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v30; // [rsp+38h] [rbp-38h]

  v8 = sub_159C470(*(_QWORD *)a2, 1, 0);
  sub_1A695F0(a1, a3, v8, a2, a4, a5);
  result = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)result <= 0x17u )
  {
    if ( (_BYTE)result != 5 )
      return result;
    v14 = *(unsigned __int16 *)(a2 + 18);
    if ( (unsigned __int16)v14 > 0x17u )
      return result;
    v15 = &loc_80A800;
    if ( !_bittest64((const __int64 *)&v15, v14) || (_WORD)v14 != 15 )
      goto LABEL_37;
    if ( (*(_BYTE *)(a2 + 17) & 4) == 0 )
      goto LABEL_16;
  }
  else
  {
    if ( (unsigned __int8)result > 0x2Fu )
      return result;
    v10 = 0x80A800000000LL;
    if ( !_bittest64(&v10, result) || (_BYTE)result != 39 || (*(_BYTE *)(a2 + 17) & 4) == 0 )
      goto LABEL_18;
  }
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v11 = *(__int64 **)(a2 - 8);
    v12 = *v11;
    if ( !*v11 )
      goto LABEL_35;
  }
  else
  {
    v11 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v12 = *v11;
    if ( !*v11 )
      goto LABEL_35;
  }
  v13 = v11[3];
  if ( *(_BYTE *)(v13 + 16) == 13 )
    return sub_1A695F0(a1, a3, v13, v12, a4, a5);
LABEL_35:
  if ( (unsigned __int8)result > 0x17u )
  {
LABEL_18:
    v17 = 0x80A800000000LL;
    if ( !_bittest64(&v17, result) )
      return result;
    result = (unsigned int)(unsigned __int8)result - 24;
    goto LABEL_20;
  }
  v14 = *(unsigned __int16 *)(a2 + 18);
LABEL_37:
  if ( (unsigned __int16)v14 > 0x17u )
    return result;
LABEL_16:
  v16 = &loc_80A800;
  result = (unsigned __int16)v14;
  if ( !_bittest64((const __int64 *)&v16, v14) )
    return result;
LABEL_20:
  if ( (_DWORD)result != 23 || (*(_BYTE *)(a2 + 17) & 4) == 0 )
    return result;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v18 = *(unsigned __int64 **)(a2 - 8);
    result = *v18;
    v25 = *v18;
    if ( !*v18 )
      return result;
  }
  else
  {
    result = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v18 = (unsigned __int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v25 = result;
    if ( !result )
      return result;
  }
  v19 = v18[3];
  if ( *(_BYTE *)(v19 + 16) != 13 )
    return result;
  v20 = *(_DWORD *)(v19 + 32);
  v28 = v20;
  if ( v20 <= 0x40 )
  {
    v30 = v20;
    v21 = v19 + 24;
    v27 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v20) & 1;
LABEL_27:
    v29 = v27;
    goto LABEL_28;
  }
  sub_16A4EF0((__int64)&v27, 1, 0);
  v21 = v19 + 24;
  v30 = v28;
  if ( v28 <= 0x40 )
    goto LABEL_27;
  sub_16A4FD0((__int64)&v29, (const void **)&v27);
  v21 = v19 + 24;
LABEL_28:
  sub_16A7E20((__int64)&v29, v21);
  v22 = (__int64 *)sub_16498A0(v19);
  v23 = sub_159C0E0(v22, (__int64)&v29);
  v24 = v23;
  if ( v30 > 0x40 && v29 )
  {
    v26 = v23;
    j_j___libc_free_0_0(v29);
    v24 = v26;
  }
  result = sub_1A695F0(a1, a3, v24, v25, a4, a5);
  if ( v28 > 0x40 )
  {
    if ( v27 )
      return j_j___libc_free_0_0(v27);
  }
  return result;
}
