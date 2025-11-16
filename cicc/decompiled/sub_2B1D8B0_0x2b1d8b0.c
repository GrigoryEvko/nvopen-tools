// Function: sub_2B1D8B0
// Address: 0x2b1d8b0
//
unsigned __int8 ***__fastcall sub_2B1D8B0(unsigned __int8 ***a1, __int64 a2)
{
  unsigned __int8 ***v2; // r14
  __int64 v4; // r12
  __int64 v5; // rax
  unsigned __int8 ***v6; // r12
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  unsigned __int8 **v11; // rbx
  unsigned __int8 **v12; // r15
  unsigned __int8 **v13; // rax
  unsigned __int8 *v14; // rsi
  unsigned __int8 *v15; // rcx
  unsigned __int8 **v16; // rbx
  unsigned __int8 **v17; // r15
  unsigned __int8 **v18; // rax
  unsigned __int8 *v19; // rsi
  unsigned __int8 *v20; // rcx
  unsigned __int8 **v21; // rbx
  unsigned __int8 **v22; // r15
  unsigned __int8 **v23; // rax
  unsigned __int8 *v24; // rsi
  unsigned __int8 *v25; // rcx
  unsigned __int8 ***result; // rax
  unsigned __int8 **v27; // rbx
  unsigned __int8 **v28; // r15
  unsigned __int8 **v29; // rax
  unsigned __int8 *v30; // rsi
  unsigned __int8 *v31; // rcx
  unsigned __int64 v32; // r12
  unsigned __int8 **v33; // r15
  unsigned __int64 v34; // r12
  unsigned __int8 **v35; // r15
  unsigned __int64 v36; // r12
  unsigned __int8 **v37; // r15
  bool v38; // r8
  bool v39; // r8

  v2 = a1;
  v4 = (a2 - (__int64)a1) >> 8;
  v5 = (a2 - (__int64)a1) >> 6;
  if ( v4 <= 0 )
  {
LABEL_55:
    if ( v5 != 2 )
    {
      if ( v5 != 3 )
      {
        if ( v5 != 1 )
          return (unsigned __int8 ***)a2;
        goto LABEL_63;
      }
      v32 = *((unsigned int *)v2 + 2);
      v33 = *v2;
      if ( v32 > 1
        && sub_2B0D880(*v2, *((unsigned int *)v2 + 2), (unsigned __int8 (__fastcall *)(_QWORD))sub_2B0D8B0)
        && sub_2B08550(v33, v32) )
      {
        return v2;
      }
      v2 += 8;
    }
    v34 = *((unsigned int *)v2 + 2);
    v35 = *v2;
    if ( v34 > 1 && sub_2B0D880(*v2, *((unsigned int *)v2 + 2), (unsigned __int8 (__fastcall *)(_QWORD))sub_2B0D8B0) )
    {
      v39 = sub_2B08550(v35, v34);
      result = v2;
      if ( v39 )
        return result;
    }
    v2 += 8;
LABEL_63:
    v36 = *((unsigned int *)v2 + 2);
    v37 = *v2;
    if ( v36 > 1 && sub_2B0D880(*v2, *((unsigned int *)v2 + 2), (unsigned __int8 (__fastcall *)(_QWORD))sub_2B0D8B0) )
    {
      v38 = sub_2B08550(v37, v36);
      result = v2;
      if ( v38 )
        return result;
    }
    return (unsigned __int8 ***)a2;
  }
  v6 = &a1[32 * v4];
  while ( 1 )
  {
    v10 = *((unsigned int *)v2 + 2);
    if ( v10 <= 1 )
      goto LABEL_3;
    v11 = *v2;
    v12 = &(*v2)[v10];
    if ( v12 != sub_2B0BF30(*v2, (__int64)v12, (unsigned __int8 (__fastcall *)(_QWORD))sub_2B0D8B0) )
      goto LABEL_3;
    v13 = v11 + 1;
    v14 = 0;
    while ( 1 )
    {
      v15 = *(v13 - 1);
      if ( (unsigned int)*v15 - 12 <= 1 )
        break;
      if ( v14 && v14 != v15 )
        goto LABEL_3;
      if ( v12 == v13 )
        return v2;
      v14 = *(v13 - 1);
LABEL_11:
      ++v13;
    }
    if ( v12 != v13 )
      goto LABEL_11;
    if ( v14 )
      return v2;
LABEL_3:
    v7 = *((unsigned int *)v2 + 18);
    if ( v7 <= 1 )
      goto LABEL_4;
    v16 = v2[8];
    v17 = &v16[v7];
    if ( v17 != sub_2B0BF30(v16, (__int64)v17, (unsigned __int8 (__fastcall *)(_QWORD))sub_2B0D8B0) )
      goto LABEL_4;
    v18 = v16 + 1;
    v19 = 0;
    while ( 2 )
    {
      v20 = *(v18 - 1);
      if ( (unsigned int)*v20 - 12 > 1 )
      {
        if ( v19 && v19 != v20 )
          goto LABEL_4;
        if ( v17 == v18 )
          return v2 + 8;
        v19 = *(v18 - 1);
        goto LABEL_20;
      }
      if ( v17 != v18 )
      {
LABEL_20:
        ++v18;
        continue;
      }
      break;
    }
    if ( v19 )
      return v2 + 8;
LABEL_4:
    v8 = *((unsigned int *)v2 + 34);
    if ( v8 <= 1 )
      goto LABEL_5;
    v21 = v2[16];
    v22 = &v21[v8];
    if ( v22 != sub_2B0BF30(v21, (__int64)v22, (unsigned __int8 (__fastcall *)(_QWORD))sub_2B0D8B0) )
      goto LABEL_5;
    v23 = v21 + 1;
    v24 = 0;
    while ( 2 )
    {
      v25 = *(v23 - 1);
      if ( (unsigned int)*v25 - 12 > 1 )
      {
        if ( v24 )
        {
          if ( v25 != v24 )
            goto LABEL_5;
        }
        else
        {
          v24 = *(v23 - 1);
        }
        if ( v22 == v23 )
          return v2 + 16;
        goto LABEL_29;
      }
      if ( v22 != v23 )
      {
LABEL_29:
        ++v23;
        continue;
      }
      break;
    }
    if ( v24 )
      return v2 + 16;
LABEL_5:
    v9 = *((unsigned int *)v2 + 50);
    if ( v9 > 1 )
    {
      v27 = v2[24];
      v28 = &v27[v9];
      if ( v28 == sub_2B0BF30(v27, (__int64)v28, (unsigned __int8 (__fastcall *)(_QWORD))sub_2B0D8B0) )
        break;
    }
LABEL_6:
    v2 += 32;
    if ( v6 == v2 )
    {
      v5 = (a2 - (__int64)v2) >> 6;
      goto LABEL_55;
    }
  }
  v29 = v27 + 1;
  v30 = 0;
  while ( 1 )
  {
    v31 = *(v29 - 1);
    if ( (unsigned int)*v31 - 12 <= 1 )
      break;
    if ( v30 && v31 != v30 )
      goto LABEL_6;
    if ( v28 == v29 )
      return v2 + 24;
    v30 = *(v29 - 1);
LABEL_38:
    ++v29;
  }
  if ( v28 != v29 )
    goto LABEL_38;
  if ( !v30 )
    goto LABEL_6;
  return v2 + 24;
}
