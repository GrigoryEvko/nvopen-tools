// Function: sub_AAE940
// Address: 0xaae940
//
__int64 __fastcall sub_AAE940(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // r13d
  unsigned int v6; // eax
  char v7; // al
  __int64 v8; // rdi
  unsigned int v9; // edx
  __int64 v10; // rdx
  unsigned int v12; // edx
  unsigned __int64 v13; // r8
  unsigned int v14; // eax
  unsigned __int64 v15; // rdx
  unsigned int v16; // eax
  unsigned int v17; // eax
  unsigned int v18; // ecx
  unsigned int v19; // eax
  unsigned int v20; // eax
  __int64 v21; // rdi
  unsigned int v25; // eax
  __int64 v27; // [rsp+0h] [rbp-70h]
  char v28; // [rsp+0h] [rbp-70h]
  unsigned int v30; // [rsp+8h] [rbp-68h]
  unsigned int v31; // [rsp+8h] [rbp-68h]
  __int64 v32; // [rsp+8h] [rbp-68h]
  unsigned int v33; // [rsp+8h] [rbp-68h]
  __int64 v34; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v35; // [rsp+18h] [rbp-58h]
  __int64 v36; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v37; // [rsp+28h] [rbp-48h]
  __int64 v38; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v39; // [rsp+38h] [rbp-38h]

  v5 = *(_DWORD *)(a2 + 8);
  v37 = v5;
  if ( v5 > 0x40 )
    sub_C43780(&v36, a2);
  else
    v36 = *(_QWORD *)a2;
  sub_C46A40(&v36, 1);
  v6 = v37;
  v37 = 0;
  v39 = v6;
  v38 = v36;
  if ( v6 <= 0x40 )
  {
    v9 = *(_DWORD *)(a2 + 8);
    if ( *(_QWORD *)a3 != v36 )
      goto LABEL_9;
LABEL_33:
    if ( v9 <= 0x40 )
    {
      _RCX = *(_QWORD *)a2;
      v25 = 64;
      __asm { tzcnt   rsi, rcx }
      if ( *(_QWORD *)a2 )
        v25 = _RSI;
      if ( v25 <= v9 )
        v9 = v25;
    }
    else
    {
      v9 = sub_C44590(a2);
    }
    sub_9691E0((__int64)&v38, v5, v9, 0, 0);
    sub_AADBC0(a1, &v38);
    if ( v39 > 0x40 )
      goto LABEL_16;
    return a1;
  }
  v27 = v36;
  v7 = sub_C43C50(&v38, a3);
  if ( v27 )
  {
    v8 = v27;
    v28 = v7;
    j_j___libc_free_0_0(v8);
    v7 = v28;
    if ( v37 > 0x40 )
    {
      if ( v36 )
      {
        j_j___libc_free_0_0(v36);
        v7 = v28;
      }
    }
  }
  v9 = *(_DWORD *)(a2 + 8);
  if ( v7 )
    goto LABEL_33;
LABEL_9:
  if ( v9 <= 0x40 )
  {
    if ( !*(_QWORD *)a2 )
      goto LABEL_11;
  }
  else if ( (unsigned int)sub_C444A0(a2) == v9 )
  {
LABEL_11:
    v10 = v5 + 1;
    goto LABEL_12;
  }
  v35 = *(_DWORD *)(a3 + 8);
  if ( v35 > 0x40 )
    sub_C43780(&v34, a3);
  else
    v34 = *(_QWORD *)a3;
  sub_C46F20(&v34, 1);
  v12 = v35;
  v35 = 0;
  v37 = v12;
  v36 = v34;
  if ( v12 > 0x40 )
  {
    sub_C43C10(&v36, a2);
    v12 = v37;
    v13 = v36;
    v37 = 0;
    v39 = v12;
    v38 = v36;
    if ( v12 > 0x40 )
    {
      v32 = v36;
      v20 = sub_C444A0(&v38);
      v12 = v20;
      if ( v32 )
      {
        v21 = v32;
        v33 = v20;
        j_j___libc_free_0_0(v21);
        v12 = v33;
        if ( v37 > 0x40 )
        {
          if ( v36 )
          {
            j_j___libc_free_0_0(v36);
            v12 = v33;
          }
        }
      }
      goto LABEL_26;
    }
  }
  else
  {
    v13 = *(_QWORD *)a2 ^ v34;
  }
  v14 = v12 - 64;
  if ( v13 )
  {
    _BitScanReverse64(&v15, v13);
    v12 = v14 + (v15 ^ 0x3F);
  }
LABEL_26:
  if ( v35 > 0x40 && v34 )
  {
    v31 = v12;
    j_j___libc_free_0_0(v34);
    v12 = v31;
  }
  v16 = *(_DWORD *)(a2 + 8);
  if ( v16 <= 0x40 )
  {
    _RSI = *(_QWORD *)a2;
    v18 = 64;
    __asm { tzcnt   rdi, rsi }
    if ( *(_QWORD *)a2 )
      v18 = _RDI;
    if ( v16 <= v18 )
      v18 = *(_DWORD *)(a2 + 8);
  }
  else
  {
    v30 = v12;
    v17 = sub_C44590(a2);
    v12 = v30;
    v18 = v17;
  }
  v19 = v5 - 1 - v12;
  if ( v19 < v18 )
    v19 = v18;
  v10 = v19 + 1;
LABEL_12:
  sub_9691E0((__int64)&v38, v5, v10, 0, 0);
  sub_9691E0((__int64)&v36, v5, 0, 0, 0);
  sub_AADC30(a1, (__int64)&v36, &v38);
  if ( v37 > 0x40 && v36 )
    j_j___libc_free_0_0(v36);
  if ( v39 > 0x40 )
  {
LABEL_16:
    if ( v38 )
      j_j___libc_free_0_0(v38);
  }
  return a1;
}
