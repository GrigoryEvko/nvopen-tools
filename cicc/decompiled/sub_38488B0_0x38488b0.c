// Function: sub_38488B0
// Address: 0x38488b0
//
unsigned __int8 *__fastcall sub_38488B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rsi
  __int16 *v9; // rdx
  __int16 v10; // ax
  __int64 v11; // rdx
  unsigned int *v12; // rcx
  unsigned int *v13; // rax
  unsigned int *v14; // rdx
  __int64 v15; // rax
  _QWORD *v16; // rdi
  unsigned __int16 *v17; // rax
  __int64 v18; // r8
  unsigned int v19; // ecx
  _QWORD *v20; // rax
  __int64 v21; // rdx
  _QWORD *v22; // r8
  __int64 v23; // r9
  __int64 v24; // rax
  unsigned __int64 v25; // rcx
  unsigned __int8 *v26; // r14
  __int128 v28; // [rsp-10h] [rbp-1B0h]
  __int64 v29; // [rsp+0h] [rbp-1A0h]
  _QWORD *v30; // [rsp+8h] [rbp-198h]
  unsigned int v31; // [rsp+1Ch] [rbp-184h]
  __int64 v32; // [rsp+30h] [rbp-170h] BYREF
  int v33; // [rsp+38h] [rbp-168h]
  __int64 v34; // [rsp+40h] [rbp-160h] BYREF
  __int64 v35; // [rsp+48h] [rbp-158h]
  __int64 v36; // [rsp+50h] [rbp-150h] BYREF
  int v37; // [rsp+58h] [rbp-148h]
  unsigned int *v38; // [rsp+60h] [rbp-140h] BYREF
  __int64 v39; // [rsp+68h] [rbp-138h]
  _OWORD v40[19]; // [rsp+70h] [rbp-130h] BYREF

  v8 = *(_QWORD *)(a2 + 80);
  v32 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v32, v8, 1);
  v9 = *(__int16 **)(a2 + 48);
  v33 = *(_DWORD *)(a2 + 72);
  v10 = *v9;
  v11 = *((_QWORD *)v9 + 1);
  LOWORD(v34) = v10;
  v35 = v11;
  if ( v10 )
  {
    if ( (unsigned __int16)(v10 - 176) > 0x34u )
    {
LABEL_5:
      v31 = word_4456340[(unsigned __int16)v34 - 1];
      goto LABEL_8;
    }
  }
  else if ( !sub_3007100((__int64)&v34) )
  {
    goto LABEL_7;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v34 )
  {
    if ( (unsigned __int16)(v34 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_5;
  }
LABEL_7:
  v31 = sub_3007130((__int64)&v34, v8);
LABEL_8:
  v38 = (unsigned int *)v40;
  v12 = (unsigned int *)v40;
  v39 = 0x1000000000LL;
  if ( !v31 )
    goto LABEL_16;
  v13 = (unsigned int *)v40;
  if ( v31 > 0x10uLL )
  {
    sub_C8D5F0((__int64)&v38, v40, v31, 0x10u, a5, a6);
    v12 = v38;
    v14 = &v38[4 * v31];
    v13 = &v38[4 * (unsigned int)v39];
    if ( v14 != v13 )
      goto LABEL_11;
  }
  else
  {
    v14 = (unsigned int *)&v40[v31];
    if ( v14 != (unsigned int *)v40 )
    {
      do
      {
LABEL_11:
        if ( v13 )
        {
          *(_QWORD *)v13 = 0;
          v13[2] = 0;
        }
        v13 += 4;
      }
      while ( v14 != v13 );
      v12 = v38;
    }
  }
  LODWORD(v39) = v31;
LABEL_16:
  v15 = *(_QWORD *)(a2 + 40);
  *(_QWORD *)v12 = *(_QWORD *)v15;
  v12[2] = *(_DWORD *)(v15 + 8);
  v16 = *(_QWORD **)(a1 + 8);
  v17 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v38 + 48LL) + 16LL * v38[2]);
  v18 = *((_QWORD *)v17 + 1);
  v19 = *v17;
  v36 = 0;
  v37 = 0;
  v20 = sub_33F17F0(v16, 51, (__int64)&v36, v19, v18);
  v22 = v20;
  v23 = v21;
  if ( v36 )
  {
    v29 = v21;
    v30 = v20;
    sub_B91220((__int64)&v36, v36);
    v23 = v29;
    v22 = v30;
  }
  v24 = 4;
  if ( v31 > 1 )
  {
    do
    {
      v25 = (unsigned __int64)v38;
      *(_QWORD *)&v38[v24] = v22;
      *(_DWORD *)(v25 + v24 * 4 + 8) = v23;
      v24 += 4;
    }
    while ( v24 != 4LL * v31 );
  }
  *((_QWORD *)&v28 + 1) = (unsigned int)v39;
  *(_QWORD *)&v28 = v38;
  v26 = sub_33FC220(*(_QWORD **)(a1 + 8), 156, (__int64)&v32, v34, v35, v23, v28);
  if ( v38 != (unsigned int *)v40 )
    _libc_free((unsigned __int64)v38);
  if ( v32 )
    sub_B91220((__int64)&v32, v32);
  return v26;
}
