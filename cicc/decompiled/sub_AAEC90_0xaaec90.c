// Function: sub_AAEC90
// Address: 0xaaec90
//
__int64 __fastcall sub_AAEC90(__int64 a1, _QWORD *a2, __int64 a3)
{
  unsigned int v5; // r14d
  unsigned int v6; // eax
  char v7; // al
  __int64 v8; // rdi
  unsigned int v9; // eax
  __int64 v10; // rdi
  unsigned int v12; // ecx
  __int64 v13; // rax
  __int64 v14; // r15
  unsigned __int64 v15; // r15
  unsigned int v16; // eax
  unsigned __int64 v17; // rcx
  int v18; // eax
  unsigned int v19; // ecx
  int v20; // r15d
  unsigned int v21; // eax
  unsigned int v22; // esi
  int v23; // edx
  unsigned int v24; // r12d
  int v25; // eax
  int v28; // eax
  unsigned int v30; // edx
  unsigned int v32; // eax
  __int64 v33; // [rsp+10h] [rbp-70h]
  char v34; // [rsp+10h] [rbp-70h]
  unsigned int v35; // [rsp+10h] [rbp-70h]
  unsigned int v36; // [rsp+10h] [rbp-70h]
  unsigned int v37; // [rsp+10h] [rbp-70h]
  unsigned int v38; // [rsp+10h] [rbp-70h]
  __int64 v40; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v41; // [rsp+28h] [rbp-58h]
  __int64 v42; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v43; // [rsp+38h] [rbp-48h]
  __int64 v44; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v45; // [rsp+48h] [rbp-38h]

  v5 = *((_DWORD *)a2 + 2);
  v43 = v5;
  if ( v5 > 0x40 )
    sub_C43780(&v42, a2);
  else
    v42 = *a2;
  sub_C46A40(&v42, 1);
  v6 = v43;
  v43 = 0;
  v45 = v6;
  v44 = v42;
  if ( v6 <= 0x40 )
  {
    v7 = *(_QWORD *)a3 == v42;
  }
  else
  {
    v33 = v42;
    v7 = sub_C43C50(&v44, a3);
    if ( v33 )
    {
      v8 = v33;
      v34 = v7;
      j_j___libc_free_0_0(v8);
      v7 = v34;
      if ( v43 > 0x40 )
      {
        if ( v42 )
        {
          j_j___libc_free_0_0(v42);
          v7 = v34;
        }
      }
    }
  }
  if ( !v7 )
  {
    v45 = *(_DWORD *)(a3 + 8);
    if ( v45 > 0x40 )
      sub_C43780(&v44, a3);
    else
      v44 = *(_QWORD *)a3;
    sub_C46F20(&v44, 1);
    v12 = *((_DWORD *)a2 + 2);
    v41 = v45;
    v13 = v44;
    v43 = v12;
    v40 = v44;
    if ( v12 > 0x40 )
    {
      sub_C43780(&v42, a2);
      v12 = v43;
      if ( v43 > 0x40 )
      {
        sub_C43C10(&v42, &v40);
        v12 = v43;
        v15 = v42;
        v43 = 0;
        v45 = v12;
        v44 = v42;
        if ( v12 > 0x40 )
        {
          v32 = sub_C444A0(&v44);
          v12 = v32;
          if ( v15 )
          {
            v38 = v32;
            j_j___libc_free_0_0(v15);
            v12 = v38;
            if ( v43 > 0x40 )
            {
              if ( v42 )
              {
                j_j___libc_free_0_0(v42);
                v12 = v38;
              }
            }
          }
LABEL_22:
          v35 = v12;
          sub_C48300(&v44, a2, v12);
          if ( v45 > 0x40 )
          {
            v28 = sub_C44630(&v44);
            v19 = v35;
            v20 = v28;
            if ( v44 )
            {
              j_j___libc_free_0_0(v44);
              v21 = *((_DWORD *)a2 + 2);
              v19 = v35;
              if ( v21 > 0x40 )
                goto LABEL_25;
              goto LABEL_44;
            }
          }
          else
          {
            v18 = sub_39FAC40(v44);
            v19 = v35;
            v20 = v18;
          }
          v21 = *((_DWORD *)a2 + 2);
          if ( v21 > 0x40 )
          {
LABEL_25:
            v36 = v19;
            v21 = sub_C44590(a2);
            v19 = v36;
            goto LABEL_26;
          }
LABEL_44:
          _RSI = *a2;
          v30 = 64;
          __asm { tzcnt   rdi, rsi }
          if ( *a2 )
            v30 = _RDI;
          if ( v21 > v30 )
            v21 = v30;
LABEL_26:
          v22 = v5 - v19;
          v23 = v5 - v19 + 1;
          v24 = v20 + (v5 - v19 > v21);
          if ( v41 > 0x40 )
          {
            v37 = v5 - v19;
            v25 = sub_C445E0(&v40);
            v23 = v22 + 1;
            v22 = v37;
          }
          else
          {
            v25 = 64;
            _RCX = ~v40;
            __asm { tzcnt   rdi, rcx }
            if ( v40 != -1 )
              v25 = _RDI;
          }
          sub_9691E0((__int64)&v44, v5, (__PAIR64__(v20 + v23, v25) - v22) >> 32, 0, 0);
          sub_9691E0((__int64)&v42, v5, v24, 0, 0);
          sub_AADC30(a1, (__int64)&v42, &v44);
          if ( v43 > 0x40 && v42 )
            j_j___libc_free_0_0(v42);
          if ( v45 > 0x40 && v44 )
            j_j___libc_free_0_0(v44);
          if ( v41 > 0x40 )
          {
            v10 = v40;
            if ( v40 )
              goto LABEL_13;
          }
          return a1;
        }
LABEL_20:
        v16 = v12 - 64;
        if ( v15 )
        {
          _BitScanReverse64(&v17, v15);
          v12 = v16 + (v17 ^ 0x3F);
        }
        goto LABEL_22;
      }
      v14 = v42;
      v13 = v40;
    }
    else
    {
      v14 = *a2;
    }
    v15 = v13 ^ v14;
    goto LABEL_20;
  }
  if ( *((_DWORD *)a2 + 2) <= 0x40u )
    v9 = sub_39FAC40(*a2);
  else
    v9 = sub_C44630(a2);
  sub_9691E0((__int64)&v44, v5, v9, 0, 0);
  sub_AADBC0(a1, &v44);
  if ( v45 > 0x40 )
  {
    v10 = v44;
    if ( v44 )
LABEL_13:
      j_j___libc_free_0_0(v10);
  }
  return a1;
}
