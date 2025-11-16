// Function: sub_33E0440
// Address: 0x33e0440
//
__int16 __fastcall sub_33E0440(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdi
  int v6; // eax
  _QWORD *v7; // rcx
  int v8; // eax
  __int16 result; // ax
  int v10; // edx
  unsigned __int64 v11; // rax
  char v12; // cl
  __int64 v13; // rdx
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  unsigned int v16; // eax
  unsigned int v17; // r14d
  unsigned __int64 v18; // rax
  unsigned int v19; // r14d
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rdx
  char v24; // cl
  unsigned __int64 v25; // rax
  __int64 v27; // rcx
  __int16 v28; // [rsp+Ch] [rbp-64h]
  __int16 v29; // [rsp+Ch] [rbp-64h]
  __int64 v30; // [rsp+10h] [rbp-60h] BYREF
  __int64 v31; // [rsp+18h] [rbp-58h] BYREF
  unsigned __int64 v32; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v33; // [rsp+28h] [rbp-48h]
  unsigned __int64 v34; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v35; // [rsp+38h] [rbp-38h]

  v5 = *(_QWORD *)(a1 + 16);
  v30 = 0;
  v31 = 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64 *, __int64 *))(*(_QWORD *)v5 + 2136LL))(
         v5,
         a2,
         &v30,
         &v31) )
  {
    v15 = sub_2E79000(*(__int64 **)(a1 + 40));
    v16 = sub_AE43A0(v15, *(_QWORD *)(v30 + 8));
    v33 = v16;
    v17 = v16;
    if ( v16 > 0x40 )
    {
      sub_C43690((__int64)&v32, 0, 0);
      v35 = v17;
      sub_C43690((__int64)&v34, 0, 0);
    }
    else
    {
      v32 = 0;
      v35 = v16;
      v34 = 0;
    }
    v18 = sub_2E79000(*(__int64 **)(a1 + 40));
    sub_9AC1B0(v30, &v32, v18, 0, 0, 0, 0, 1);
    v19 = v33;
    if ( v33 <= 0x40 )
    {
      _RAX = ~v32;
      __asm { tzcnt   rcx, rax }
      if ( v32 == -1 )
      {
        v21 = v31;
        v22 = 0x80000000LL;
        goto LABEL_21;
      }
    }
    else
    {
      LODWORD(_RCX) = sub_C445E0((__int64)&v32);
    }
    if ( !(_DWORD)_RCX )
    {
      if ( v35 > 0x40 && v34 )
      {
        j_j___libc_free_0_0(v34);
        v19 = v33;
      }
      if ( v19 > 0x40 && v32 )
        j_j___libc_free_0_0(v32);
      goto LABEL_2;
    }
    v21 = v31;
    v22 = 0x80000000LL;
    if ( (unsigned int)_RCX <= 0x1E )
    {
      _BitScanReverse64((unsigned __int64 *)&_RCX, 1LL << _RCX);
      v22 = 0x8000000000000000LL >> ((unsigned __int8)_RCX ^ 0x3Fu);
    }
LABEL_21:
    v23 = v21 | v22;
    v24 = -1;
    if ( (v23 & -(__int64)v23) != 0 )
    {
      _BitScanReverse64(&v25, v23 & -(__int64)v23);
      v24 = 63 - (v25 ^ 0x3F);
    }
    LOBYTE(result) = v24;
    HIBYTE(result) = 1;
    if ( v35 > 0x40 && v34 )
    {
      v28 = result;
      j_j___libc_free_0_0(v34);
      v19 = v33;
      result = v28;
    }
    if ( v19 > 0x40 && v32 )
    {
      v29 = result;
      j_j___libc_free_0_0(v32);
      return v29;
    }
    return result;
  }
LABEL_2:
  v6 = *(_DWORD *)(a2 + 24);
  if ( v6 == 15 || v6 == 39 )
  {
    v10 = *(_DWORD *)(a2 + 96);
    v11 = 0;
  }
  else
  {
    if ( !sub_33E0400(a1, a2, a3) )
      return 0;
    v7 = *(_QWORD **)(a2 + 40);
    v8 = *(_DWORD *)(*v7 + 24LL);
    if ( v8 != 15 && v8 != 39 )
      return 0;
    v10 = *(_DWORD *)(*v7 + 96LL);
    v27 = *(_QWORD *)(v7[5] + 96LL);
    v11 = *(_QWORD *)(v27 + 24);
    if ( *(_DWORD *)(v27 + 32) > 0x40u )
      v11 = *(_QWORD *)v11;
  }
  if ( v10 == 0x80000000 )
    return 0;
  v12 = -1;
  v13 = v11
      | (1LL << *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 48LL) + 8LL)
                         + 40LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 48LL) + 32LL) + v10)
                         + 16));
  v14 = v13 & -v13;
  if ( v14 )
  {
    _BitScanReverse64(&v14, v14);
    v12 = 63 - (v14 ^ 0x3F);
  }
  LOBYTE(result) = v12;
  HIBYTE(result) = 1;
  return result;
}
