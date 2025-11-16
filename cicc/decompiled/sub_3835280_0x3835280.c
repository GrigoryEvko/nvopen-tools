// Function: sub_3835280
// Address: 0x3835280
//
__int64 *__fastcall sub_3835280(__int64 a1, __int64 *a2)
{
  unsigned __int64 v2; // rbx
  __int64 v4; // rax
  unsigned __int16 v5; // r13
  __int64 v6; // rax
  unsigned int v7; // eax
  __int64 v8; // r13
  __int64 v9; // r10
  int v10; // ecx
  unsigned int v11; // edi
  __int64 v12; // rax
  int v13; // r11d
  __int64 v14; // r10
  unsigned __int64 v15; // r11
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  _QWORD *v18; // rax
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rcx
  __int64 v23; // rax
  _BYTE *v24; // rdx
  __int64 *v25; // r12
  int v27; // eax
  unsigned __int64 v28; // [rsp+8h] [rbp-188h]
  __int64 v29; // [rsp+20h] [rbp-170h]
  int *v30; // [rsp+28h] [rbp-168h]
  __int64 v31; // [rsp+28h] [rbp-168h]
  int v32; // [rsp+3Ch] [rbp-154h] BYREF
  unsigned __int16 v33; // [rsp+40h] [rbp-150h] BYREF
  __int64 v34; // [rsp+48h] [rbp-148h]
  _BYTE *v35; // [rsp+50h] [rbp-140h] BYREF
  __int64 v36; // [rsp+58h] [rbp-138h]
  _BYTE v37[304]; // [rsp+60h] [rbp-130h] BYREF

  v4 = a2[6];
  v5 = *(_WORD *)v4;
  v6 = *(_QWORD *)(v4 + 8);
  v33 = v5;
  v34 = v6;
  if ( v5 )
  {
    if ( (unsigned __int16)(v5 - 176) <= 0x34u )
    {
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    }
    v7 = word_4456340[v5 - 1];
  }
  else
  {
    if ( sub_3007100((__int64)&v33) )
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
    v7 = sub_3007130((__int64)&v33, (__int64)a2);
  }
  v35 = v37;
  v36 = 0x1000000000LL;
  if ( v7 )
  {
    v8 = 0;
    v29 = 40LL * v7;
    while ( 1 )
    {
      v32 = sub_375D5B0(a1, *(_QWORD *)(a2[5] + v8), *(_QWORD *)(a2[5] + v8 + 8));
      v30 = sub_3805BC0(a1 + 712, &v32);
      sub_37593F0(a1, v30);
      if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
      {
        v9 = a1 + 520;
        v10 = 7;
      }
      else
      {
        v22 = *(unsigned int *)(a1 + 528);
        v9 = *(_QWORD *)(a1 + 520);
        if ( !(_DWORD)v22 )
          goto LABEL_19;
        v10 = v22 - 1;
      }
      v11 = v10 & (37 * *v30);
      v12 = v9 + 24LL * v11;
      v13 = *(_DWORD *)v12;
      if ( *v30 != *(_DWORD *)v12 )
        break;
LABEL_12:
      v14 = *(_QWORD *)(v12 + 8);
      v15 = *(unsigned int *)(v12 + 16) | v2 & 0xFFFFFFFF00000000LL;
      v16 = (unsigned int)v36;
      v2 = v15;
      v17 = (unsigned int)v36 + 1LL;
      if ( v17 > HIDWORD(v36) )
      {
        v28 = v15;
        v31 = v14;
        sub_C8D5F0((__int64)&v35, v37, v17, 0x10u, v20, v21);
        v16 = (unsigned int)v36;
        v15 = v28;
        v14 = v31;
      }
      v18 = &v35[16 * v16];
      v8 += 40;
      *v18 = v14;
      v18[1] = v15;
      v19 = (unsigned int)(v36 + 1);
      LODWORD(v36) = v36 + 1;
      if ( v29 == v8 )
      {
        v24 = v35;
        goto LABEL_22;
      }
    }
    v27 = 1;
    while ( v13 != -1 )
    {
      v20 = (unsigned int)(v27 + 1);
      v11 = v10 & (v27 + v11);
      v12 = v9 + 24LL * v11;
      v13 = *(_DWORD *)v12;
      if ( *v30 == *(_DWORD *)v12 )
        goto LABEL_12;
      v27 = v20;
    }
    if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
    {
      v23 = 192;
    }
    else
    {
      v22 = *(unsigned int *)(a1 + 528);
LABEL_19:
      v23 = 24 * v22;
    }
    v12 = v9 + v23;
    goto LABEL_12;
  }
  v24 = v37;
  v19 = 0;
LABEL_22:
  v25 = sub_33EC210(*(_QWORD **)(a1 + 8), a2, (__int64)v24, v19);
  if ( v35 != v37 )
    _libc_free((unsigned __int64)v35);
  return v25;
}
