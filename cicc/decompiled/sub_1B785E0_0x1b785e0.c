// Function: sub_1B785E0
// Address: 0x1b785e0
//
__int64 __fastcall sub_1B785E0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned __int64 *v10; // rax
  bool v11; // zf
  __int64 v12; // rcx
  __int64 *v13; // r8
  double v14; // xmm4_8
  double v15; // xmm5_8
  __int64 v16; // rdx
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v19; // r15
  __int64 v20; // r14
  unsigned __int64 *v21; // r12
  unsigned __int64 *v22; // rbx
  __int64 v23; // rdi
  __int64 v25; // [rsp+18h] [rbp-4A8h]
  __int64 v26; // [rsp+38h] [rbp-488h]
  __int64 *v27; // [rsp+40h] [rbp-480h] BYREF
  char v28; // [rsp+48h] [rbp-478h]
  __int64 v29; // [rsp+50h] [rbp-470h] BYREF
  _BYTE *v30; // [rsp+58h] [rbp-468h]
  __int64 v31; // [rsp+60h] [rbp-460h]
  _BYTE v32[128]; // [rsp+68h] [rbp-458h] BYREF
  __int64 v33; // [rsp+E8h] [rbp-3D8h]
  __int64 v34; // [rsp+F0h] [rbp-3D0h]
  unsigned __int64 *v35; // [rsp+F8h] [rbp-3C8h] BYREF
  unsigned int v36; // [rsp+100h] [rbp-3C0h]
  unsigned __int64 v37[2]; // [rsp+3F8h] [rbp-C8h] BYREF
  _BYTE v38[184]; // [rsp+408h] [rbp-B8h] BYREF

  v30 = v32;
  v31 = 0x1000000000LL;
  v10 = (unsigned __int64 *)&v35;
  v29 = a1;
  v33 = 0;
  v34 = 1;
  do
  {
    *v10 = -4;
    v10 += 3;
  }
  while ( v10 != v37 );
  v11 = *(_BYTE *)(a2 + 1) == 0;
  v37[0] = (unsigned __int64)v38;
  v37[1] = 0x1000000000LL;
  if ( v11 )
    v25 = (__int64)sub_1B76A40(&v29, a2, a3, a4, a5, a6, a7, a8, a9, a10);
  else
    v25 = sub_1B757D0((__int64)&v29, a2, a3, a4, a5, a6, a7, a8, a9, a10);
  v16 = (unsigned int)v31;
  while ( (_DWORD)v16 )
  {
    while ( 1 )
    {
      v12 = (unsigned int)v16;
      v16 = (unsigned int)(v16 - 1);
      v17 = *(_QWORD *)&v30[8 * v12 - 8];
      LODWORD(v31) = v16;
      v18 = *(unsigned int *)(v17 + 8);
      if ( !(_DWORD)v18 )
        break;
      v26 = *(unsigned int *)(v17 + 8);
      v19 = 0;
      while ( 1 )
      {
        a2 = (__int64)&v29;
        v20 = *(_QWORD *)(v17 + 8 * (v19 - v18));
        sub_1B76990((__int64)&v27, &v29, v20, a3, a4, a5, a6, v14, v15, a9, a10);
        if ( v28 )
        {
          v13 = v27;
        }
        else
        {
          a2 = v20;
          v13 = sub_1B76A40(&v29, v20, a3, a4, a5, a6, v14, v15, a9, a10);
        }
        if ( (__int64 *)v20 != v13 )
        {
          a2 = (unsigned int)v19;
          sub_1630830(v17, v19, (unsigned __int8 *)v13, *(double *)a3.m128_u64, a4, a5, a6, v14, v15, a9, a10);
        }
        if ( v26 == ++v19 )
          break;
        v18 = *(unsigned int *)(v17 + 8);
      }
      v16 = (unsigned int)v31;
      if ( !(_DWORD)v31 )
        goto LABEL_17;
    }
  }
LABEL_17:
  if ( (_BYTE *)v37[0] != v38 )
    _libc_free(v37[0]);
  if ( (v34 & 1) != 0 )
  {
    v22 = v37;
    v21 = (unsigned __int64 *)&v35;
  }
  else
  {
    v21 = v35;
    if ( !v36 )
    {
LABEL_32:
      j___libc_free_0(v21);
      goto LABEL_28;
    }
    v22 = &v35[3 * v36];
  }
  do
  {
    if ( *v21 != -8 && *v21 != -4 )
    {
      v23 = v21[2];
      if ( v23 )
        sub_16307F0(v23, a2, v16, v12, (__int64)v13, a3, a4, a5, a6, v14, v15, a9, a10);
    }
    v21 += 3;
  }
  while ( v22 != v21 );
  if ( (v34 & 1) == 0 )
  {
    v21 = v35;
    goto LABEL_32;
  }
LABEL_28:
  if ( v30 != v32 )
    _libc_free((unsigned __int64)v30);
  return v25;
}
