// Function: sub_1254190
// Address: 0x1254190
//
__int64 __fastcall sub_1254190(__int64 a1, char *a2, unsigned __int64 a3)
{
  unsigned int v3; // r12d
  unsigned int v4; // r13d
  __int64 result; // rax
  unsigned int v6; // edx
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // r12
  bool v9; // cc
  unsigned int v10; // r15d
  __int64 v11; // rdx
  unsigned int v12; // edx
  unsigned __int64 v13; // r12
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v16; // [rsp+8h] [rbp-48h]
  char *v17; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v18; // [rsp+18h] [rbp-38h]

  *(_DWORD *)(a1 + 8) = 1;
  *(_QWORD *)a1 = 0;
  v3 = (a3 << 6) / 0x13 + 2;
  *(_BYTE *)(a1 + 12) = 0;
  sub_C47AB0((__int64)&v15, v3, a2, a3, 0xAu);
  v4 = v16;
  if ( *a2 == 45 )
  {
    v10 = v16 + 1;
    v11 = 1LL << ((unsigned __int8)v16 - 1);
    result = v15;
    if ( v16 <= 0x40 )
    {
      if ( (v15 & v11) != 0 )
      {
        if ( v16 )
        {
          v12 = v16 - 63;
          result = ~(v15 << (64 - (unsigned __int8)v16));
          if ( v15 << (64 - (unsigned __int8)v16) != -1 )
          {
            _BitScanReverse64((unsigned __int64 *)&result, result);
            result ^= 0x3FuLL;
            v12 = v10 - result;
          }
        }
        else
        {
          v12 = 1;
        }
      }
      else
      {
        v12 = 1;
        if ( v15 )
        {
          _BitScanReverse64(&v14, v15);
          result = v14 ^ 0x3F;
          v12 = 65 - result;
        }
      }
      if ( v3 <= v12 )
        goto LABEL_45;
    }
    else
    {
      if ( (*(_QWORD *)(v15 + 8LL * ((v16 - 1) >> 6)) & v11) != 0 )
        v12 = v10 - sub_C44500((__int64)&v15);
      else
        v12 = v10 - sub_C444A0((__int64)&v15);
      if ( v3 <= v12 )
      {
        v18 = v4;
        goto LABEL_29;
      }
    }
    if ( !v12 )
      v12 = 1;
    sub_C44740((__int64)&v17, (char **)&v15, v12);
    if ( v16 > 0x40 && v15 )
      j_j___libc_free_0_0(v15);
    result = (__int64)v17;
    v4 = v18;
    v15 = (unsigned __int64)v17;
    v16 = v18;
    if ( v18 > 0x40 )
    {
LABEL_29:
      result = (__int64)sub_C43780((__int64)&v17, (const void **)&v15);
      v9 = *(_DWORD *)(a1 + 8) <= 0x40u;
      v4 = v18;
      v18 = 0;
      v13 = (unsigned __int64)v17;
      if ( v9 )
        goto LABEL_46;
      goto LABEL_30;
    }
LABEL_45:
    v13 = v15;
    v9 = *(_DWORD *)(a1 + 8) <= 0x40u;
    v18 = 0;
    v17 = (char *)v15;
    if ( v9 )
      goto LABEL_46;
LABEL_30:
    if ( *(_QWORD *)a1 )
    {
      result = j_j___libc_free_0_0(*(_QWORD *)a1);
      v9 = v18 <= 0x40;
      *(_QWORD *)a1 = v13;
      *(_DWORD *)(a1 + 8) = v4;
      *(_BYTE *)(a1 + 12) = 0;
      if ( v9 )
        goto LABEL_11;
      goto LABEL_9;
    }
LABEL_46:
    *(_QWORD *)a1 = v13;
    *(_DWORD *)(a1 + 8) = v4;
    *(_BYTE *)(a1 + 12) = 0;
    goto LABEL_11;
  }
  if ( v16 > 0x40 )
  {
    v6 = v4 - sub_C444A0((__int64)&v15);
    if ( v3 <= v6 )
    {
      v18 = v4;
      goto LABEL_22;
    }
LABEL_16:
    if ( !v6 )
      v6 = 1;
    sub_C44740((__int64)&v17, (char **)&v15, v6);
    if ( v16 > 0x40 && v15 )
      j_j___libc_free_0_0(v15);
    result = (__int64)v17;
    v4 = v18;
    v15 = (unsigned __int64)v17;
    v16 = v18;
    if ( v18 <= 0x40 )
      goto LABEL_6;
LABEL_22:
    result = (__int64)sub_C43780((__int64)&v17, (const void **)&v15);
    v9 = *(_DWORD *)(a1 + 8) <= 0x40u;
    v4 = v18;
    v18 = 0;
    v8 = (unsigned __int64)v17;
    if ( v9 )
      goto LABEL_23;
    goto LABEL_7;
  }
  result = v15;
  v6 = 0;
  if ( v15 )
  {
    _BitScanReverse64(&v7, v15);
    result = v7 ^ 0x3F;
    v6 = 64 - result;
  }
  if ( v3 > v6 )
    goto LABEL_16;
LABEL_6:
  v8 = v15;
  v9 = *(_DWORD *)(a1 + 8) <= 0x40u;
  v18 = 0;
  v17 = (char *)v15;
  if ( v9 )
  {
LABEL_23:
    *(_QWORD *)a1 = v8;
    *(_DWORD *)(a1 + 8) = v4;
    *(_BYTE *)(a1 + 12) = 1;
    goto LABEL_11;
  }
LABEL_7:
  if ( !*(_QWORD *)a1 )
    goto LABEL_23;
  result = j_j___libc_free_0_0(*(_QWORD *)a1);
  v9 = v18 <= 0x40;
  *(_QWORD *)a1 = v8;
  *(_DWORD *)(a1 + 8) = v4;
  *(_BYTE *)(a1 + 12) = 1;
  if ( v9 )
    goto LABEL_11;
LABEL_9:
  if ( v17 )
    result = j_j___libc_free_0_0(v17);
LABEL_11:
  if ( v16 > 0x40 )
  {
    if ( v15 )
      return j_j___libc_free_0_0(v15);
  }
  return result;
}
