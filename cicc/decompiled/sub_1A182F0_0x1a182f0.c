// Function: sub_1A182F0
// Address: 0x1a182f0
//
__int64 __fastcall sub_1A182F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 *v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rbx
  __int64 *v15; // rax
  __int64 *v16; // rsi
  __int64 v17; // rdx
  __int64 *v18; // rbx
  __int64 *v19; // r12
  double v20; // xmm4_8
  double v21; // xmm5_8
  __int64 v22; // rbx
  __int64 v23; // r14
  __int64 *v24; // rax
  __int64 *v25; // r15
  __int64 v26; // rdx
  __int64 v27; // r15
  __int64 v28; // r14
  __int64 v29; // rax
  __int64 v30; // rcx
  _QWORD *v31; // r12
  unsigned __int8 v32; // al
  __int64 v34; // rax
  __int64 v35; // [rsp+0h] [rbp-9C0h]
  unsigned __int8 v36; // [rsp+Eh] [rbp-9B2h]
  unsigned __int8 v37; // [rsp+Fh] [rbp-9B1h]
  _BYTE v38[16]; // [rsp+10h] [rbp-9B0h] BYREF
  __int64 v39; // [rsp+20h] [rbp-9A0h] BYREF
  __int64 *v40; // [rsp+28h] [rbp-998h]
  __int64 *v41; // [rsp+30h] [rbp-990h]
  unsigned int v42; // [rsp+38h] [rbp-988h]
  unsigned int v43; // [rsp+3Ch] [rbp-984h]
  int v44; // [rsp+40h] [rbp-980h]
  __int64 v45; // [rsp+760h] [rbp-260h] BYREF
  unsigned int v46; // [rsp+768h] [rbp-258h]
  unsigned int v47; // [rsp+76Ch] [rbp-254h]
  _BYTE v48[592]; // [rsp+770h] [rbp-250h] BYREF

  sub_1A0F500((__int64)v38, a2, a3);
  v14 = *(_QWORD *)(a1 + 80);
  if ( v14 )
    v14 -= 24;
  v15 = v40;
  if ( v41 != v40 )
    goto LABEL_4;
  v16 = &v40[v43];
  if ( v40 == v16 )
  {
LABEL_57:
    if ( v43 < v42 )
    {
      ++v43;
      *v16 = v14;
      ++v39;
      goto LABEL_45;
    }
LABEL_4:
    v16 = (__int64 *)v14;
    sub_16CCBA0((__int64)&v39, v14);
    if ( !(_BYTE)v17 )
      goto LABEL_5;
LABEL_45:
    v34 = v46;
    if ( v46 >= v47 )
    {
      v16 = (__int64 *)v48;
      sub_16CD150((__int64)&v45, v48, 0, 8, v12, v13);
      v34 = v46;
    }
    v17 = v45;
    *(_QWORD *)(v45 + 8 * v34) = v14;
    ++v46;
    if ( (*(_BYTE *)(a1 + 18) & 1) == 0 )
      goto LABEL_6;
    goto LABEL_48;
  }
  v11 = 0;
  while ( 1 )
  {
    v17 = *v15;
    if ( v14 == *v15 )
      break;
    if ( v17 == -2 )
      v11 = v15;
    if ( v16 == ++v15 )
    {
      if ( !v11 )
        goto LABEL_57;
      *v11 = v14;
      --v44;
      ++v39;
      goto LABEL_45;
    }
  }
LABEL_5:
  if ( (*(_BYTE *)(a1 + 18) & 1) == 0 )
  {
LABEL_6:
    v18 = *(__int64 **)(a1 + 88);
    v19 = &v18[5 * *(_QWORD *)(a1 + 96)];
    goto LABEL_7;
  }
LABEL_48:
  sub_15E08E0(a1, (__int64)v16);
  v18 = *(__int64 **)(a1 + 88);
  v19 = &v18[5 * *(_QWORD *)(a1 + 96)];
  if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
  {
    sub_15E08E0(a1, (__int64)v16);
    v18 = *(__int64 **)(a1 + 88);
  }
LABEL_7:
  while ( v19 != v18 )
  {
    v16 = v18;
    v18 += 5;
    sub_1A11830((__int64)v38, (__int64)v16);
  }
  do
  {
    sub_1A174E0((__int64)v38, *(double *)a4.m128_u64, a5, a6, (__int64)v16, v17, (__int64)v11, v12, v13);
    v16 = (__int64 *)a1;
    v37 = sub_1A17850((__int64)v38, a1);
  }
  while ( v37 );
  v22 = *(_QWORD *)(a1 + 80);
  v35 = a1 + 72;
  if ( a1 + 72 != v22 )
  {
    while ( 1 )
    {
      v23 = 0;
      v24 = v40;
      if ( v22 )
        v23 = v22 - 24;
      if ( v41 == v40 )
      {
        v25 = &v40[v43];
        if ( v40 == v25 )
        {
          v26 = (__int64)v40;
        }
        else
        {
          do
          {
            if ( v23 == *v24 )
              break;
            ++v24;
          }
          while ( v25 != v24 );
          v26 = (__int64)&v40[v43];
        }
LABEL_35:
        while ( (__int64 *)v26 != v24 )
        {
          if ( (unsigned __int64)*v24 < 0xFFFFFFFFFFFFFFFELL )
            goto LABEL_17;
          ++v24;
        }
        if ( v25 == v24 )
          goto LABEL_37;
LABEL_18:
        v27 = *(_QWORD *)(v23 + 48);
        v28 = v23 + 40;
        if ( v28 != v27 )
        {
          while ( 1 )
          {
            v29 = v27;
            v27 = *(_QWORD *)(v27 + 8);
            if ( !*(_BYTE *)(*(_QWORD *)(v29 - 24) + 8LL) )
              goto LABEL_20;
            v30 = (unsigned int)*(unsigned __int8 *)(v29 - 8) - 25;
            if ( (unsigned int)v30 <= 9 )
              goto LABEL_20;
            v31 = (_QWORD *)(v29 - 24);
            v32 = sub_1A13400((__int64)v38, (_QWORD *)(v29 - 24), a4, a5, a6, a7, v20, v21, a10, a11, v26, v30);
            if ( !v32 )
              goto LABEL_20;
            v36 = v32;
            v37 = sub_1AE9990(v31, 0);
            if ( v37 )
            {
              sub_15F20C0(v31);
LABEL_20:
              if ( v28 == v27 )
                break;
            }
            else
            {
              v37 = v36;
              if ( v28 == v27 )
                break;
            }
          }
        }
        v22 = *(_QWORD *)(v22 + 8);
        if ( v35 == v22 )
          break;
      }
      else
      {
        v25 = &v41[v42];
        v24 = sub_16CC9F0((__int64)&v39, v23);
        if ( v23 == *v24 )
        {
          if ( v41 == v40 )
            v26 = (__int64)&v41[v43];
          else
            v26 = (__int64)&v41[v42];
          goto LABEL_35;
        }
        if ( v41 == v40 )
        {
          v24 = &v41[v43];
          v26 = (__int64)v24;
          goto LABEL_35;
        }
        v26 = v42;
        v24 = &v41[v42];
LABEL_17:
        if ( v25 != v24 )
          goto LABEL_18;
LABEL_37:
        sub_1AEBFA0(v23);
        v37 = 1;
        v22 = *(_QWORD *)(v22 + 8);
        if ( v35 == v22 )
          break;
      }
    }
  }
  sub_1A0F740((__int64)v38);
  return v37;
}
