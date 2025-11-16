// Function: sub_C79480
// Address: 0xc79480
//
unsigned __int64 *__fastcall sub_C79480(unsigned __int64 *a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v8; // rbx
  unsigned int v9; // edx
  bool v10; // al
  unsigned int v11; // ebx
  bool v12; // al
  unsigned int v13; // edx
  unsigned __int64 v14; // rcx
  const void *v15; // rcx
  int v16; // eax
  bool v17; // al
  unsigned int v18; // eax
  unsigned __int64 v19; // rdx
  unsigned int v20; // r10d
  unsigned int v21; // esi
  char v22; // r8
  bool v23; // cc
  unsigned int v24; // eax
  unsigned __int64 v25; // rdi
  __int64 v26; // rdx
  unsigned __int64 v27; // rax
  __int64 v28; // rcx
  unsigned int v29; // eax
  unsigned int v31; // [rsp+8h] [rbp-B8h]
  unsigned int v32; // [rsp+Ch] [rbp-B4h]
  const void *v33; // [rsp+10h] [rbp-B0h]
  char v34; // [rsp+10h] [rbp-B0h]
  const void **v35; // [rsp+18h] [rbp-A8h]
  __int64 v36; // [rsp+20h] [rbp-A0h] BYREF
  unsigned int v37; // [rsp+28h] [rbp-98h]
  const void *v38; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v39; // [rsp+38h] [rbp-88h]
  const void *v40; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v41; // [rsp+48h] [rbp-78h]
  unsigned __int64 v42; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v43; // [rsp+58h] [rbp-68h]
  unsigned __int64 v44; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v45; // [rsp+68h] [rbp-58h]
  const void *v46; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v47; // [rsp+78h] [rbp-48h]
  unsigned __int64 v48; // [rsp+80h] [rbp-40h]
  int v49; // [rsp+88h] [rbp-38h]

  v8 = *(unsigned int *)(a2 + 8);
  v35 = (const void **)(a1 + 2);
  *((_DWORD *)a1 + 2) = v8;
  if ( (unsigned int)v8 > 0x40 )
  {
    sub_C43690((__int64)a1, 0, 0);
    *((_DWORD *)a1 + 6) = v8;
    sub_C43690((__int64)v35, 0, 0);
    v9 = *(_DWORD *)(a2 + 8);
    if ( !v9 )
      goto LABEL_64;
  }
  else
  {
    *a1 = 0;
    *((_DWORD *)a1 + 6) = v8;
    a1[2] = 0;
    v9 = *(_DWORD *)(a2 + 8);
    if ( !v9 )
      goto LABEL_53;
  }
  if ( v9 <= 0x40 )
    v10 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v9) == *(_QWORD *)a2;
  else
    v10 = (unsigned int)sub_C445E0(a2) == v9;
  if ( v10
    || (v11 = *(_DWORD *)(a3 + 8)) == 0
    || (v11 <= 0x40
      ? (v12 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v11) == *(_QWORD *)a3)
      : (v12 = v11 == (unsigned int)sub_C445E0(a3)),
        v12) )
  {
LABEL_64:
    v8 = *((unsigned int *)a1 + 2);
    if ( (unsigned int)v8 > 0x40 )
    {
      memset((void *)*a1, -1, 8 * (((unsigned __int64)(unsigned int)v8 + 63) >> 6));
      v8 = *((unsigned int *)a1 + 2);
      v26 = *a1;
LABEL_54:
      v27 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v8;
      if ( (_DWORD)v8 )
      {
        if ( (unsigned int)v8 > 0x40 )
        {
          v28 = (unsigned int)((unsigned __int64)(v8 + 63) >> 6) - 1;
          *(_QWORD *)(v26 + 8 * v28) &= v27;
          v29 = *((_DWORD *)a1 + 6);
          if ( v29 <= 0x40 )
          {
LABEL_57:
            a1[2] = 0;
            return a1;
          }
LABEL_61:
          memset((void *)a1[2], 0, 8 * (((unsigned __int64)v29 + 63) >> 6));
          return a1;
        }
      }
      else
      {
        v27 = 0;
      }
      *a1 = v26 & v27;
      v29 = *((_DWORD *)a1 + 6);
      if ( v29 <= 0x40 )
        goto LABEL_57;
      goto LABEL_61;
    }
LABEL_53:
    *a1 = -1;
    v26 = -1;
    goto LABEL_54;
  }
  v37 = *(_DWORD *)(a3 + 24);
  if ( v37 > 0x40 )
    sub_C43780((__int64)&v36, (const void **)(a3 + 16));
  else
    v36 = *(_QWORD *)(a3 + 16);
  v13 = *(_DWORD *)(a2 + 8);
  v47 = v13;
  if ( v13 > 0x40 )
  {
    sub_C43780((__int64)&v46, (const void **)a2);
    v13 = v47;
    if ( v47 > 0x40 )
    {
      sub_C43D10((__int64)&v46);
      v13 = v47;
      v15 = v46;
      goto LABEL_16;
    }
    v14 = (unsigned __int64)v46;
  }
  else
  {
    v14 = *(_QWORD *)a2;
  }
  v15 = (const void *)(~v14 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v13));
  if ( !v13 )
    v15 = 0;
LABEL_16:
  v39 = v13;
  v38 = v15;
  if ( v37 <= 0x40 )
  {
    v17 = v36 == 0;
  }
  else
  {
    v31 = v13;
    v32 = v37;
    v33 = v15;
    v16 = sub_C444A0((__int64)&v36);
    v15 = v33;
    v13 = v31;
    v17 = v32 == v16;
  }
  if ( v17 )
  {
    v41 = v13;
    if ( v13 <= 0x40 )
    {
      v40 = v15;
      goto LABEL_21;
    }
    sub_C43780((__int64)&v40, &v38);
  }
  else
  {
    sub_C4A1D0((__int64)&v40, (__int64)&v38, (__int64)&v36);
  }
  v13 = v41;
  if ( v41 > 0x40 )
  {
    v13 = sub_C444A0((__int64)&v40);
    goto LABEL_23;
  }
  v15 = v40;
LABEL_21:
  v18 = v13 - 64;
  if ( v15 )
  {
    _BitScanReverse64(&v19, (unsigned __int64)v15);
    v13 = v18 + (v19 ^ 0x3F);
  }
LABEL_23:
  v20 = *((_DWORD *)a1 + 2);
  v21 = v20 - v13;
  if ( v20 != v20 - v13 )
  {
    if ( v21 <= 0x3F && v20 <= 0x40 )
    {
      v43 = *((_DWORD *)a1 + 2);
      v22 = a4;
      *a1 |= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v13) << v21;
      goto LABEL_27;
    }
    sub_C43C90(a1, v21, v20);
    v21 = *((_DWORD *)a1 + 2);
  }
  v43 = v21;
  v22 = a4;
  if ( v21 > 0x40 )
  {
    sub_C43780((__int64)&v42, (const void **)a1);
    v22 = a4;
    goto LABEL_28;
  }
LABEL_27:
  v42 = *a1;
LABEL_28:
  v45 = *((_DWORD *)a1 + 6);
  if ( v45 > 0x40 )
  {
    v34 = v22;
    sub_C43780((__int64)&v44, v35);
    v22 = v34;
  }
  else
  {
    v44 = a1[2];
  }
  sub_C6FD10((__int64)&v46, (__int64)&v42, a2, a3, v22);
  if ( *((_DWORD *)a1 + 2) > 0x40u && *a1 )
    j_j___libc_free_0_0(*a1);
  v23 = *((_DWORD *)a1 + 6) <= 0x40u;
  *a1 = (unsigned __int64)v46;
  v24 = v47;
  v47 = 0;
  *((_DWORD *)a1 + 2) = v24;
  if ( v23 || (v25 = a1[2]) == 0 )
  {
    a1[2] = v48;
    *((_DWORD *)a1 + 6) = v49;
  }
  else
  {
    j_j___libc_free_0_0(v25);
    v23 = v47 <= 0x40;
    a1[2] = v48;
    *((_DWORD *)a1 + 6) = v49;
    if ( !v23 && v46 )
      j_j___libc_free_0_0(v46);
  }
  if ( v45 > 0x40 && v44 )
    j_j___libc_free_0_0(v44);
  if ( v43 > 0x40 && v42 )
    j_j___libc_free_0_0(v42);
  if ( v41 > 0x40 && v40 )
    j_j___libc_free_0_0(v40);
  if ( v39 > 0x40 && v38 )
    j_j___libc_free_0_0(v38);
  if ( v37 > 0x40 && v36 )
    j_j___libc_free_0_0(v36);
  return a1;
}
