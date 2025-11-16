// Function: sub_123E5A0
// Address: 0x123e5a0
//
__int64 __fastcall sub_123E5A0(__int64 a1, __int64 a2, unsigned __int64 a3, unsigned int a4)
{
  __int64 v4; // r15
  unsigned __int64 v7; // rsi
  unsigned int v8; // r12d
  int v10; // r15d
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // rcx
  const __m128i *v16; // r8
  const __m128i *v17; // rcx
  const __m128i *v18; // r15
  const __m128i **v19; // rax
  __int64 v20; // rdx
  _QWORD *v21; // r9
  signed __int64 v22; // r8
  _BYTE *v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // rdx
  int v26; // eax
  __int64 v27; // [rsp-8h] [rbp-E8h]
  __int64 v28; // [rsp+8h] [rbp-D8h]
  const __m128i *v29; // [rsp+10h] [rbp-D0h]
  __int64 v30; // [rsp+10h] [rbp-D0h]
  __int64 v31; // [rsp+10h] [rbp-D0h]
  int v32; // [rsp+10h] [rbp-D0h]
  int v33; // [rsp+18h] [rbp-C8h]
  const __m128i *v34; // [rsp+18h] [rbp-C8h]
  _QWORD *v35; // [rsp+18h] [rbp-C8h]
  __int64 v36; // [rsp+18h] [rbp-C8h]
  __int64 v37; // [rsp+18h] [rbp-C8h]
  __int64 v38; // [rsp+18h] [rbp-C8h]
  unsigned __int64 v40; // [rsp+28h] [rbp-B8h]
  int v41; // [rsp+30h] [rbp-B0h] BYREF
  int v42; // [rsp+34h] [rbp-ACh] BYREF
  __int64 v43; // [rsp+38h] [rbp-A8h] BYREF
  __m128i v44; // [rsp+40h] [rbp-A0h] BYREF
  const __m128i **v45; // [rsp+50h] [rbp-90h] BYREF
  __int64 v46; // [rsp+58h] [rbp-88h]
  const __m128i *v47; // [rsp+60h] [rbp-80h] BYREF
  const __m128i *v48; // [rsp+68h] [rbp-78h]
  const __m128i *v49; // [rsp+70h] [rbp-70h]
  __int64 v50[2]; // [rsp+80h] [rbp-60h] BYREF
  _QWORD v51[2]; // [rsp+90h] [rbp-50h] BYREF
  char v52; // [rsp+A0h] [rbp-40h]
  char v53; // [rsp+A1h] [rbp-3Fh]

  v4 = a1 + 176;
  v40 = *(_QWORD *)(a1 + 232);
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  LOWORD(v41) = 0;
  v7 = 16;
  v44 = 0u;
  LOBYTE(v42) = 0;
  v45 = &v47;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
    || (v7 = 12, (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here"))
    || (v7 = (unsigned __int64)&v44, (unsigned __int8)sub_1212200(a1, &v44))
    || (v7 = 4, (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' here"))
    || (v7 = (unsigned __int64)&v41, (unsigned __int8)sub_1211B70(a1, &v41))
    || (v7 = 4, (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' here"))
    || (v7 = (unsigned __int64)&v42, (unsigned __int8)sub_1211F50(a1, &v42)) )
  {
LABEL_2:
    v8 = 1;
    goto LABEL_3;
  }
  while ( *(_DWORD *)(a1 + 240) == 4 )
  {
    v26 = sub_1205200(v4);
    *(_DWORD *)(a1 + 240) = v26;
    if ( v26 == 448 )
    {
      v7 = (unsigned __int64)&v47;
      if ( (unsigned __int8)sub_123AAC0(a1, &v47) )
        goto LABEL_2;
    }
    else
    {
      if ( v26 != 451 )
      {
        v53 = 1;
        v7 = *(_QWORD *)(a1 + 232);
        v52 = 3;
        v50[0] = (__int64)"expected optional variable summary field";
        sub_11FD800(v4, v7, (__int64)v50, 1);
        goto LABEL_2;
      }
      v7 = (unsigned __int64)&v45;
      if ( (unsigned __int8)sub_123A2F0(a1, (__int64)&v45) )
        goto LABEL_2;
    }
  }
  v7 = 13;
  if ( (unsigned __int8)sub_120AFE0(a1, 13, "expected ')' here") )
    goto LABEL_2;
  v10 = v42;
  v33 = v41;
  v11 = sub_22077B0(72);
  v14 = v11;
  if ( v11 )
  {
    *(_DWORD *)(v11 + 8) = 2;
    *(_QWORD *)(v11 + 16) = 0;
    *(_DWORD *)(v11 + 12) = v33;
    v15 = (unsigned int)v46;
    *(_QWORD *)v11 = &unk_49D9770;
    *(_QWORD *)(v11 + 24) = 0;
    *(_QWORD *)(v11 + 32) = 0;
    *(_QWORD *)(v11 + 40) = v11 + 56;
    *(_QWORD *)(v11 + 48) = 0;
    if ( (_DWORD)v15 )
    {
      v38 = v11;
      sub_1205B50(v11 + 40, (char **)&v45, v11, v15, v12, v13);
      v14 = v38;
    }
    *(_QWORD *)(v14 + 56) = 0;
    *(_DWORD *)(v14 + 64) = v10;
    *(_QWORD *)v14 = &unk_49D97D0;
  }
  v16 = v49;
  v28 = v14;
  v17 = v48;
  v18 = v47;
  v49 = 0;
  *(__m128i *)(v14 + 24) = v44;
  v34 = v16;
  v29 = v17;
  v48 = 0;
  v47 = 0;
  v19 = (const __m128i **)sub_22077B0(24);
  v20 = v28;
  if ( !v19 )
  {
    v21 = *(_QWORD **)(v28 + 56);
    v22 = (char *)v34 - (char *)v18;
    *(_QWORD *)(v28 + 56) = 0;
    if ( !v21 )
    {
LABEL_26:
      if ( v18 )
      {
        v37 = v20;
        j_j___libc_free_0(v18, v22);
        v20 = v37;
      }
      goto LABEL_28;
    }
LABEL_23:
    if ( *v21 )
    {
      v30 = v22;
      v35 = v21;
      j_j___libc_free_0(*v21, v21[2] - *v21);
      v20 = v28;
      v22 = v30;
      v21 = v35;
    }
    v31 = v20;
    v36 = v22;
    j_j___libc_free_0(v21, 24);
    v20 = v31;
    v22 = v36;
    goto LABEL_26;
  }
  v21 = *(_QWORD **)(v28 + 56);
  *v19 = v18;
  v19[2] = v34;
  v19[1] = v29;
  *(_QWORD *)(v28 + 56) = v19;
  if ( v21 )
  {
    v22 = 0;
    v18 = 0;
    goto LABEL_23;
  }
LABEL_28:
  v23 = *(_BYTE **)a2;
  v43 = v20;
  v24 = *(_QWORD *)(a2 + 8);
  v50[0] = (__int64)v51;
  v32 = v41 & 0xF;
  sub_12060D0(v50, v23, (__int64)&v23[v24]);
  v7 = (unsigned __int64)v50;
  v8 = sub_123DE00(a1, v50, a3, v32, a4, &v43, v40);
  v25 = v27;
  if ( (_QWORD *)v50[0] != v51 )
  {
    v7 = v51[0] + 1LL;
    j_j___libc_free_0(v50[0], v51[0] + 1LL);
  }
  if ( v43 )
    (*(void (__fastcall **)(__int64, unsigned __int64, __int64))(*(_QWORD *)v43 + 8LL))(v43, v7, v25);
LABEL_3:
  if ( v47 )
  {
    v7 = (char *)v49 - (char *)v47;
    j_j___libc_free_0(v47, (char *)v49 - (char *)v47);
  }
  if ( v45 != &v47 )
    _libc_free(v45, v7);
  return v8;
}
