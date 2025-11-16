// Function: sub_11948A0
// Address: 0x11948a0
//
__int64 __fastcall sub_11948A0(unsigned __int8 *a1, __m128i *a2)
{
  unsigned __int64 v3; // r13
  _BYTE *v4; // rax
  __int64 v5; // rsi
  unsigned __int64 v6; // rax
  unsigned int v7; // edx
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // r15
  unsigned int v14; // esi
  unsigned __int64 v15; // rdx
  __int64 v16; // rax
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int64 *v23; // [rsp+8h] [rbp-88h]
  unsigned __int64 v24; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v25; // [rsp+18h] [rbp-78h]
  unsigned __int64 v26; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v27; // [rsp+28h] [rbp-68h]
  __int64 v28; // [rsp+30h] [rbp-60h]
  unsigned int v29; // [rsp+38h] [rbp-58h]
  unsigned __int64 v30; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v31; // [rsp+48h] [rbp-48h]
  unsigned __int64 v32; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v33; // [rsp+58h] [rbp-38h]

  if ( *a1 == 54 )
  {
    if ( sub_B448F0((__int64)a1) )
    {
      LODWORD(v3) = 0;
      if ( sub_B44900((__int64)a1) )
        return (unsigned int)v3;
    }
    v5 = *((_QWORD *)a1 - 4);
  }
  else
  {
    LODWORD(v3) = 0;
    if ( sub_B44E60((__int64)a1) )
      return (unsigned int)v3;
    v4 = (_BYTE *)*((_QWORD *)a1 - 8);
    v5 = *((_QWORD *)a1 - 4);
    if ( *v4 == 54 && v5 == *((_QWORD *)v4 - 4) )
    {
      LODWORD(v3) = 1;
      sub_B448B0((__int64)a1, 1);
      return (unsigned int)v3;
    }
  }
  sub_9AC330((__int64)&v26, v5, 0, a2);
  LODWORD(v3) = v27;
  v31 = v27;
  if ( v27 <= 0x40 )
  {
    v6 = v26;
    v7 = v27;
LABEL_6:
    v8 = ~v6;
    if ( v7 )
    {
      v9 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v7) & v8;
      v10 = (unsigned int)(v3 - 1);
      v25 = v7;
      v24 = v9;
      if ( v10 < v9 )
        goto LABEL_8;
    }
    else
    {
      v9 = 0;
    }
    v10 = v9;
    goto LABEL_8;
  }
  v10 = v27 - 1;
  sub_C43780((__int64)&v30, (const void **)&v26);
  v7 = v31;
  if ( v31 <= 0x40 )
  {
    v6 = v30;
    goto LABEL_6;
  }
  sub_C43D10((__int64)&v30);
  LODWORD(v3) = v31;
  v25 = v31;
  v24 = v30;
  if ( v31 > 0x40 )
  {
    v23 = (unsigned __int64 *)v30;
    LODWORD(v3) = v3 - sub_C444A0((__int64)&v24);
    if ( (unsigned int)v3 > 0x40 || (v3 = *v23, v10 < *v23) )
    {
      if ( !v23 )
        goto LABEL_8;
      v3 = v10;
    }
    v10 = v3;
    j_j___libc_free_0_0(v23);
    goto LABEL_8;
  }
  if ( v30 <= v10 )
    v10 = v30;
LABEL_8:
  sub_9AC330((__int64)&v30, *((_QWORD *)a1 - 8), 0, a2);
  if ( *a1 != 54 )
  {
    if ( v31 > 0x40 )
    {
      _RAX = (unsigned int)sub_C445E0((__int64)&v30);
    }
    else
    {
      _RDX = ~v30;
      __asm { tzcnt   rax, rdx }
      _RAX = (int)_RAX;
      if ( v30 == -1 )
        _RAX = 64;
    }
    LOBYTE(v3) = _RAX >= v10;
    sub_B448B0((__int64)a1, _RAX >= v10);
    goto LABEL_13;
  }
  LODWORD(v3) = sub_B448F0((__int64)a1);
  if ( (_BYTE)v3 )
  {
    LODWORD(v3) = 0;
  }
  else
  {
    if ( v31 > 0x40 )
    {
      v20 = (unsigned int)sub_C44500((__int64)&v30);
    }
    else if ( v31 )
    {
      v20 = 64;
      if ( v30 << (64 - (unsigned __int8)v31) != -1 )
      {
        _BitScanReverse64(&v21, ~(v30 << (64 - (unsigned __int8)v31)));
        v20 = (int)(v21 ^ 0x3F);
      }
    }
    else
    {
      v20 = 0;
    }
    if ( v20 >= v10 )
    {
      LODWORD(v3) = 1;
      sub_B447F0(a1, 1);
    }
  }
  if ( sub_B44900((__int64)a1) )
    goto LABEL_13;
  v14 = v31;
  v15 = v30;
  v16 = 1LL << ((unsigned __int8)v31 - 1);
  if ( v31 > 0x40 )
  {
    if ( (*(_QWORD *)(v30 + 8LL * ((v31 - 1) >> 6)) & v16) != 0 )
    {
      v17 = (unsigned int)sub_C44500((__int64)&v30);
      goto LABEL_42;
    }
LABEL_55:
    v14 = v33;
    v15 = v32;
    v22 = 1LL << ((unsigned __int8)v33 - 1);
    if ( v33 > 0x40 )
    {
      if ( (*(_QWORD *)(v32 + 8LL * ((v33 - 1) >> 6)) & v22) != 0 )
      {
        v17 = (unsigned int)sub_C44500((__int64)&v32);
        goto LABEL_42;
      }
    }
    else if ( (v22 & v32) != 0 )
    {
      goto LABEL_39;
    }
    v17 = 1;
    goto LABEL_42;
  }
  if ( (v16 & v30) == 0 )
    goto LABEL_55;
LABEL_39:
  if ( !v14 )
    goto LABEL_43;
  v17 = 64;
  v18 = ~(v15 << (64 - (unsigned __int8)v14));
  if ( v18 )
  {
    _BitScanReverse64(&v19, v18);
    v17 = (int)(v19 ^ 0x3F);
  }
LABEL_42:
  if ( v17 <= v10 )
  {
LABEL_43:
    if ( (unsigned int)sub_9AF8B0(
                         *((_QWORD *)a1 - 8),
                         a2->m128i_i64[0],
                         0,
                         a2[2].m128i_i64[0],
                         a2[2].m128i_i64[1],
                         a2[1].m128i_i64[1],
                         1) <= v10 )
      goto LABEL_13;
  }
  LODWORD(v3) = 1;
  sub_B44850(a1, 1);
LABEL_13:
  if ( v33 > 0x40 && v32 )
    j_j___libc_free_0_0(v32);
  if ( v31 > 0x40 && v30 )
    j_j___libc_free_0_0(v30);
  if ( v29 > 0x40 && v28 )
    j_j___libc_free_0_0(v28);
  if ( v27 > 0x40 && v26 )
    j_j___libc_free_0_0(v26);
  return (unsigned int)v3;
}
