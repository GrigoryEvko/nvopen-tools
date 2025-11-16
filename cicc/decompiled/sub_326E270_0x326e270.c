// Function: sub_326E270
// Address: 0x326e270
//
__int64 __fastcall sub_326E270(__int64 a1, __m128i *a2, __int64 *a3, __int64 a4, char a5, unsigned __int8 a6)
{
  __int64 *v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rcx
  __int32 v14; // r9d
  __int64 v15; // r8
  __int64 v16; // r14
  unsigned int v17; // r12d
  __int64 v19; // rax
  __int64 v20; // rsi
  unsigned __int16 *v21; // rax
  unsigned __int16 v22; // bx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rdx
  int *v26; // rax
  int *v27; // rsi
  __int64 v28; // rcx
  __int64 v29; // rdx
  int *v30; // rdx
  int *v31; // rax
  __int64 v32; // rdx
  int *v33; // rcx
  __int64 v34; // rdx
  int *v35; // rdx
  signed __int64 v36; // rdx
  __int64 v38; // [rsp+10h] [rbp-50h]
  __int32 v39; // [rsp+1Ch] [rbp-44h]
  unsigned __int16 v40; // [rsp+20h] [rbp-40h] BYREF
  __int64 v41; // [rsp+28h] [rbp-38h]

  v10 = *(__int64 **)(a1 + 8);
  if ( a6 )
    v10 = *(__int64 **)a1;
  v11 = *v10;
  v12 = *(_QWORD *)(a1 + 32);
  if ( a5 )
  {
    v13 = *(_QWORD *)(a1 + 16);
    v14 = *(_DWORD *)(v13 + 8);
    v15 = *(_QWORD *)v13;
  }
  else
  {
    v19 = *(_QWORD *)(a1 + 24);
    v14 = *(_DWORD *)(v19 + 8);
    v15 = *(_QWORD *)v19;
    v12 = *(_QWORD *)(a1 + 40);
  }
  v16 = *(_QWORD *)v12;
  if ( !a6 )
  {
    v14 = *(_DWORD *)(v12 + 8);
    v16 = v15;
    v15 = *(_QWORD *)v12;
  }
  if ( *(_DWORD *)(v16 + 24) != 165 )
    return 0;
  v38 = v15;
  v39 = v14;
  if ( !(unsigned __int8)sub_33CF8D0(v11, v16) )
    return 0;
  v20 = a6;
  v17 = sub_326DCC0(
          *(_QWORD *)(a1 + 48),
          a6,
          **(_QWORD **)(a1 + 56),
          (_QWORD *)v16,
          v38,
          v39,
          *(_QWORD *)(*(_QWORD *)(a1 + 64) + 8LL),
          a2,
          a3,
          (__int64 *)a4);
  if ( !(_BYTE)v17 )
    return 0;
  v21 = *(unsigned __int16 **)(v16 + 48);
  v22 = *v21;
  v23 = *((_QWORD *)v21 + 1);
  v40 = v22;
  v41 = v23;
  if ( v22 )
  {
    if ( (unsigned __int16)(v22 - 176) <= 0x34u )
    {
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    }
    v24 = word_4456340[v22 - 1];
  }
  else
  {
    if ( sub_3007100((__int64)&v40) )
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
    v24 = (unsigned int)sub_3007130((__int64)&v40, v20);
  }
  v25 = 4 * v24;
  v26 = *(int **)(v16 + 96);
  v27 = &v26[(unsigned __int64)v25 / 4];
  v28 = v25 >> 2;
  v29 = v25 >> 4;
  if ( !v29 )
  {
LABEL_37:
    if ( v28 != 2 )
    {
      if ( v28 != 3 )
      {
        if ( v28 != 1 )
          goto LABEL_25;
        goto LABEL_40;
      }
      if ( *v26 < 0 )
        goto LABEL_24;
      ++v26;
    }
    if ( *v26 < 0 )
      goto LABEL_24;
    ++v26;
LABEL_40:
    if ( *v26 < 0 )
      goto LABEL_24;
LABEL_25:
    v31 = *(int **)a4;
    v32 = 4LL * *(unsigned int *)(a4 + 8);
    v33 = (int *)(*(_QWORD *)a4 + v32);
    v34 = v32 >> 4;
    if ( v34 )
    {
      v35 = &v31[4 * v34];
      while ( *v31 >= 0 )
      {
        if ( v31[1] < 0 )
        {
          LOBYTE(v17) = v33 == v31 + 1;
          return v17;
        }
        if ( v31[2] < 0 )
        {
          LOBYTE(v17) = v33 == v31 + 2;
          return v17;
        }
        if ( v31[3] < 0 )
        {
          LOBYTE(v17) = v33 == v31 + 3;
          return v17;
        }
        v31 += 4;
        if ( v35 == v31 )
          goto LABEL_45;
      }
      goto LABEL_32;
    }
LABEL_45:
    v36 = (char *)v33 - (char *)v31;
    if ( (char *)v33 - (char *)v31 != 8 )
    {
      if ( v36 != 12 )
      {
        if ( v36 != 4 )
          return v17;
        goto LABEL_48;
      }
      if ( *v31 < 0 )
      {
        LOBYTE(v17) = v31 == v33;
        return v17;
      }
      ++v31;
    }
    if ( *v31 < 0 )
    {
LABEL_32:
      LOBYTE(v17) = v33 == v31;
      return v17;
    }
    ++v31;
LABEL_48:
    if ( *v31 >= 0 )
      return v17;
    goto LABEL_32;
  }
  v30 = &v26[4 * v29];
  while ( *v26 >= 0 )
  {
    if ( v26[1] < 0 )
    {
      ++v26;
      break;
    }
    if ( v26[2] < 0 )
    {
      v26 += 2;
      break;
    }
    if ( v26[3] < 0 )
    {
      v26 += 3;
      break;
    }
    v26 += 4;
    if ( v30 == v26 )
    {
      v28 = v27 - v26;
      goto LABEL_37;
    }
  }
LABEL_24:
  if ( v27 == v26 )
    goto LABEL_25;
  return v17;
}
