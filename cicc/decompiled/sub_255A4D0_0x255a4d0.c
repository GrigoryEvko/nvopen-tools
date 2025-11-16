// Function: sub_255A4D0
// Address: 0x255a4d0
//
__int64 __fastcall sub_255A4D0(__int64 a1, __int64 a2)
{
  __m128i *v2; // r13
  char v4; // al
  unsigned int v5; // r8d
  unsigned __int64 v6; // rdi
  __int64 *v7; // rdi
  char v8; // al
  __int64 *v9; // r15
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 *v13; // rbx
  unsigned __int8 *v14; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  unsigned __int64 v21; // rdi
  __int64 *v22; // rax
  char v23; // al
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 *v34; // [rsp+8h] [rbp-A8h]
  __int64 *v36; // [rsp+18h] [rbp-98h]
  unsigned int v37; // [rsp+18h] [rbp-98h]
  unsigned int v38; // [rsp+18h] [rbp-98h]
  __int64 *v39; // [rsp+20h] [rbp-90h] BYREF
  __int64 v40; // [rsp+28h] [rbp-88h]
  _BYTE v41[32]; // [rsp+30h] [rbp-80h] BYREF
  _BYTE *v42; // [rsp+50h] [rbp-60h] BYREF
  __int64 v43; // [rsp+58h] [rbp-58h]
  _BYTE v44[80]; // [rsp+60h] [rbp-50h] BYREF

  v2 = (__m128i *)(a1 + 72);
  LODWORD(v42) = 50;
  v4 = sub_2516400(a2, (__m128i *)(a1 + 72), (__int64)&v42, 1, 1, 0);
  v5 = 1;
  if ( v4 )
    return v5;
  v39 = (__int64 *)v41;
  v40 = 0x400000000LL;
  v6 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL;
  if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
    v6 = *(_QWORD *)(v6 + 24);
  v7 = (__int64 *)sub_BD5C60(v6);
  v8 = *(_BYTE *)(a1 + 97);
  if ( (v8 & 3) == 3 )
  {
    v24 = sub_A778C0(v7, 50, 0);
    sub_255A480((__int64)&v39, v24, v25, v26, v27, v28);
  }
  else if ( (v8 & 2) != 0 )
  {
    v16 = sub_A778C0(v7, 51, 0);
    sub_255A480((__int64)&v39, v16, v17, v18, v19, v20);
  }
  else if ( (v8 & 1) != 0 )
  {
    v29 = sub_A778C0(v7, 78, 0);
    sub_255A480((__int64)&v39, v29, v30, v31, v32, v33);
  }
  v9 = v39;
  v10 = 8LL * (unsigned int)v40;
  v34 = &v39[(unsigned __int64)v10 / 8];
  v11 = v10 >> 3;
  v12 = v10 >> 5;
  if ( v12 )
  {
    v36 = &v39[4 * v12];
    while ( 1 )
    {
      LODWORD(v42) = sub_A71AE0(v9);
      if ( !(unsigned __int8)sub_2516400(a2, v2, (__int64)&v42, 1, 1, 0) )
        goto LABEL_15;
      v13 = v9++;
      LODWORD(v42) = sub_A71AE0(v9);
      if ( !(unsigned __int8)sub_2516400(a2, v2, (__int64)&v42, 1, 1, 0) )
        goto LABEL_15;
      v9 = v13 + 2;
      LODWORD(v42) = sub_A71AE0(v13 + 2);
      if ( !(unsigned __int8)sub_2516400(a2, v2, (__int64)&v42, 1, 1, 0) )
        goto LABEL_15;
      v9 = v13 + 3;
      LODWORD(v42) = sub_A71AE0(v13 + 3);
      if ( !(unsigned __int8)sub_2516400(a2, v2, (__int64)&v42, 1, 1, 0) )
        goto LABEL_15;
      v9 = v13 + 4;
      if ( v36 == v13 + 4 )
      {
        v11 = v34 - v9;
        break;
      }
    }
  }
  if ( v11 != 2 )
  {
    if ( v11 != 3 )
    {
      v5 = 1;
      if ( v11 != 1 )
        goto LABEL_19;
      goto LABEL_33;
    }
    LODWORD(v42) = sub_A71AE0(v9);
    if ( !(unsigned __int8)sub_2516400(a2, v2, (__int64)&v42, 1, 1, 0) )
      goto LABEL_15;
    ++v9;
  }
  LODWORD(v42) = sub_A71AE0(v9);
  if ( !(unsigned __int8)sub_2516400(a2, v2, (__int64)&v42, 1, 1, 0) )
    goto LABEL_15;
  ++v9;
LABEL_33:
  LODWORD(v42) = sub_A71AE0(v9);
  v23 = sub_2516400(a2, v2, (__int64)&v42, 1, 1, 0);
  v5 = 1;
  if ( !v23 )
  {
LABEL_15:
    v5 = 1;
    if ( v34 != v9 )
    {
      sub_2515E10(a2, v2->m128i_i64, (__int64)dword_438A680, 3);
      if ( (*(_BYTE *)(a1 + 97) & 2) != 0 )
      {
        LODWORD(v42) = 77;
        sub_2515E10(a2, v2->m128i_i64, (__int64)&v42, 1);
      }
      v14 = (unsigned __int8 *)sub_250D070(v2);
      v5 = 1;
      if ( (unsigned int)*v14 - 12 > 1 )
      {
        v43 = 0x400000000LL;
        v42 = v44;
        v21 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL;
        if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
          v21 = *(_QWORD *)(v21 + 24);
        v22 = (__int64 *)sub_BD5C60(v21);
        sub_25499E0(a1, a2, v22, (__int64)&v42);
        v5 = 1;
        if ( (_DWORD)v43 )
          v5 = sub_2516380(a2, v2->m128i_i64, (__int64)v42, (unsigned int)v43, 0);
        if ( v42 != v44 )
        {
          v38 = v5;
          _libc_free((unsigned __int64)v42);
          v5 = v38;
        }
      }
    }
  }
LABEL_19:
  if ( v39 != (__int64 *)v41 )
  {
    v37 = v5;
    _libc_free((unsigned __int64)v39);
    return v37;
  }
  return v5;
}
