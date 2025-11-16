// Function: sub_9714E0
// Address: 0x9714e0
//
__int64 __fastcall sub_9714E0(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4)
{
  unsigned __int8 *v7; // r14
  __int64 v8; // r14
  __int64 v10; // rsi
  char v11; // r14
  __int64 v12; // rax
  __int64 v13; // rdx
  char v14; // cl
  unsigned __int64 v15; // r8
  unsigned int v16; // r14d
  signed __int64 v17; // rdx
  __int64 v18; // rax
  unsigned int v19; // edx
  unsigned __int64 v20; // rsi
  __int64 v21; // rax
  int v22; // eax
  __int64 v23; // r8
  int v24; // eax
  signed __int64 *v25; // r8
  signed __int64 v26; // rdx
  int v27; // eax
  char v28; // cl
  __int64 v29; // rsi
  unsigned __int64 v30; // rax
  int v31; // eax
  unsigned __int64 v32; // rax
  signed __int64 *v33; // [rsp+8h] [rbp-58h]
  __int64 *v34; // [rsp+10h] [rbp-50h]
  unsigned __int64 v35; // [rsp+10h] [rbp-50h]
  int v36; // [rsp+18h] [rbp-48h]
  __int64 v37; // [rsp+18h] [rbp-48h]
  __int64 v38; // [rsp+20h] [rbp-40h] BYREF
  __int64 v39; // [rsp+28h] [rbp-38h]

  LODWORD(v39) = *(_DWORD *)(a3 + 8);
  if ( (unsigned int)v39 > 0x40 )
    sub_C43780(&v38, a3);
  else
    v38 = *(_QWORD *)a3;
  v7 = sub_96E2C0((unsigned __int8 *)a1, (__int64)&v38, (unsigned __int64)a4);
  if ( (unsigned int)v39 > 0x40 && v38 )
    j_j___libc_free_0_0(v38);
  if ( v7 )
  {
    v8 = sub_971220((__int64)v7, a2, (__int64)a4);
    if ( v8 )
      return v8;
  }
  v10 = *(_QWORD *)(a1 + 8);
  v11 = sub_AE5020(a4, v10);
  v12 = sub_9208B0((__int64)a4, v10);
  v39 = v13;
  v38 = v12;
  if ( (_BYTE)v13 )
  {
LABEL_14:
    v8 = sub_96E500((unsigned __int8 *)a1, a2, (__int64)a4);
    if ( v8 )
      return v8;
    v19 = *(_DWORD *)(a3 + 8);
    v20 = *(_QWORD *)a3;
    v21 = 1LL << ((unsigned __int8)v19 - 1);
    if ( v19 > 0x40 )
    {
      v34 = *(__int64 **)a3;
      v36 = *(_DWORD *)(a3 + 8);
      if ( (*(_QWORD *)(v20 + 8LL * ((v19 - 1) >> 6)) & v21) != 0 )
        v22 = sub_C44500(a3);
      else
        v22 = sub_C444A0(a3);
      if ( (unsigned int)(v36 + 1 - v22) <= 0x40 )
      {
        v23 = *v34;
        return sub_970C30(a1, a2, v23, a4);
      }
      return v8;
    }
    if ( (v21 & v20) != 0 )
    {
      if ( !v19 )
      {
        v23 = 0;
        return sub_970C30(a1, a2, v23, a4);
      }
      v27 = 64;
      v28 = 64 - v19;
      v29 = v20 << (64 - (unsigned __int8)v19);
      if ( v29 != -1 )
      {
        _BitScanReverse64(&v30, ~v29);
        v27 = v30 ^ 0x3F;
      }
      if ( v19 + 1 - v27 > 0x40 )
        return v8;
    }
    else
    {
      if ( v20 )
      {
        _BitScanReverse64(&v32, v20);
        if ( (unsigned int)v32 == 0x3F )
          return v8;
      }
      v23 = 0;
      if ( !v19 )
        return sub_970C30(a1, a2, v23, a4);
      v28 = 64 - v19;
      v29 = v20 << (64 - (unsigned __int8)v19);
    }
    v23 = v29 >> v28;
    return sub_970C30(a1, a2, v23, a4);
  }
  v14 = v11;
  v15 = *(_QWORD *)a3;
  v16 = *(_DWORD *)(a3 + 8);
  v17 = (((unsigned __int64)(v12 + 7) >> 3) + (1LL << v14) - 1) >> v14 << v14;
  if ( v16 > 0x40 )
  {
    v33 = *(signed __int64 **)a3;
    v35 = (((unsigned __int64)(v12 + 7) >> 3) + (1LL << v14) - 1) >> v14 << v14;
    v37 = *(_QWORD *)(v15 + 8LL * ((v16 - 1) >> 6)) & (1LL << ((unsigned __int8)v16 - 1));
    if ( v37 )
    {
      v31 = sub_C44500(a3);
      v26 = v35;
      v25 = v33;
      if ( v16 + 1 - v31 > 0x40 )
      {
LABEL_24:
        if ( v37 )
          goto LABEL_14;
        return sub_ACADE0(a2);
      }
    }
    else
    {
      v24 = sub_C444A0(a3);
      v25 = v33;
      v26 = v35;
      if ( v16 + 1 - v24 > 0x40 )
        goto LABEL_24;
    }
    if ( v26 > *v25 )
      goto LABEL_14;
    return sub_ACADE0(a2);
  }
  v18 = 0;
  if ( v16 )
    v18 = (__int64)(v15 << (64 - (unsigned __int8)v16)) >> (64 - (unsigned __int8)v16);
  if ( v17 > v18 )
    goto LABEL_14;
  return sub_ACADE0(a2);
}
