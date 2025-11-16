// Function: sub_20748F0
// Address: 0x20748f0
//
__int64 *__fastcall sub_20748F0(__int64 a1, __int64 a2, char a3, __m128i a4, __m128i a5, __m128i a6)
{
  __int64 v7; // rax
  int v8; // edx
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 *v11; // r13
  __int64 *v12; // r14
  __int64 v13; // r8
  __int64 v14; // rdx
  __int64 *v15; // r13
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  int v20; // edx
  int v21; // r8d
  __int64 v22; // r10
  int v23; // eax
  unsigned int v24; // edx
  __int64 v25; // rax
  __int64 v26; // rax
  _QWORD *v27; // r14
  __int64 *v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  int v31; // edx
  __int64 v32; // r8
  __int64 v33; // r13
  int v34; // r14d
  __int64 *result; // rax
  __int64 v36; // rsi
  int v37; // eax
  _QWORD *v38; // rcx
  __int64 v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // [rsp+10h] [rbp-100h]
  int v45; // [rsp+10h] [rbp-100h]
  int v46; // [rsp+18h] [rbp-F8h]
  __int64 v47; // [rsp+18h] [rbp-F8h]
  __int64 *v48; // [rsp+20h] [rbp-F0h]
  __int128 v49; // [rsp+20h] [rbp-F0h]
  __int64 *v50; // [rsp+20h] [rbp-F0h]
  __int128 v51; // [rsp+30h] [rbp-E0h]
  __int64 v53; // [rsp+48h] [rbp-C8h]
  __int64 v54; // [rsp+48h] [rbp-C8h]
  __int64 v55; // [rsp+80h] [rbp-90h] BYREF
  int v56; // [rsp+88h] [rbp-88h]
  __int64 v57; // [rsp+90h] [rbp-80h] BYREF
  __int64 v58; // [rsp+98h] [rbp-78h]
  __int64 v59[4]; // [rsp+A0h] [rbp-70h] BYREF
  __int128 v60; // [rsp+C0h] [rbp-50h] BYREF
  __int64 v61; // [rsp+D0h] [rbp-40h]

  v7 = *(_QWORD *)a1;
  v8 = *(_DWORD *)(a1 + 536);
  v55 = 0;
  v56 = v8;
  if ( v7 )
  {
    if ( &v55 != (__int64 *)(v7 + 48) )
    {
      v9 = *(_QWORD *)(v7 + 48);
      v55 = v9;
      if ( v9 )
        sub_1623A60((__int64)&v55, v9, 2);
    }
  }
  v10 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v11 = *(__int64 **)(a2 - 24 * v10);
  v12 = *(__int64 **)(a2 + 24 * (1 - v10));
  v13 = *(_QWORD *)(a2 + 24 * (2 - v10));
  if ( a3 )
  {
    v48 = *(__int64 **)(a2 + 24 * (2 - v10));
    *(_QWORD *)&v51 = sub_20685E0(a1, *(__int64 **)(a2 + 24 * (1 - v10)), a4, a5, a6);
    *((_QWORD *)&v51 + 1) = v14;
    v15 = sub_20685E0(a1, v11, a4, a5, a6);
    v53 = v16;
    *(_QWORD *)&v49 = sub_20685E0(a1, v48, a4, a5, a6);
    *((_QWORD *)&v49 + 1) = v17;
    v18 = v15[5] + 16LL * (unsigned int)v53;
    LOBYTE(v17) = *(_BYTE *)v18;
    v19 = *(_QWORD *)(v18 + 8);
    LOBYTE(v57) = v17;
    v58 = v19;
LABEL_7:
    v46 = sub_1D172F0(*(_QWORD *)(a1 + 552), (unsigned int)v57, v58);
    goto LABEL_8;
  }
  v38 = *(_QWORD **)(v13 + 24);
  if ( *(_DWORD *)(v13 + 32) > 0x40u )
    v38 = (_QWORD *)*v38;
  v46 = (int)v38;
  v45 = (int)v38;
  v50 = *(__int64 **)(a2 + 24 * (3 - v10));
  *(_QWORD *)&v51 = sub_20685E0(a1, *(__int64 **)(a2 + 24 * (1 - v10)), a4, a5, a6);
  *((_QWORD *)&v51 + 1) = v39;
  v15 = sub_20685E0(a1, v11, a4, a5, a6);
  v53 = v40;
  *(_QWORD *)&v49 = sub_20685E0(a1, v50, a4, a5, a6);
  *((_QWORD *)&v49 + 1) = v41;
  v42 = v15[5] + 16LL * (unsigned int)v53;
  LOBYTE(v41) = *(_BYTE *)v42;
  v43 = *(_QWORD *)(v42 + 8);
  LOBYTE(v57) = v41;
  v58 = v43;
  if ( !v45 )
    goto LABEL_7;
LABEL_8:
  memset(v59, 0, 24);
  sub_14A8180(a2, v59, 0);
  if ( (_BYTE)v57 )
  {
    v20 = sub_2045180(v57);
  }
  else
  {
    v44 = *(_QWORD *)(*(_QWORD *)(a1 + 552) + 32LL);
    v37 = sub_1F58D40((__int64)&v57);
    v21 = (unsigned int)v59;
    v22 = v44;
    v20 = v37;
  }
  v60 = (unsigned __int64)v12;
  v23 = 0;
  v24 = (unsigned int)(v20 + 7) >> 3;
  LOBYTE(v61) = 0;
  if ( v12 )
  {
    v25 = *v12;
    if ( *(_BYTE *)(*v12 + 8) == 16 )
      v25 = **(_QWORD **)(v25 + 16);
    v23 = *(_DWORD *)(v25 + 8) >> 8;
  }
  HIDWORD(v61) = v23;
  v26 = sub_1E0B8E0(v22, 2u, v24, v46, v21, 0, v60, v61, 1u, 0, 0);
  v27 = *(_QWORD **)(a1 + 552);
  v47 = v26;
  v28 = sub_2051C20((__int64 *)a1, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
  v30 = sub_1D2C870(v27, (__int64)v28, v29, (__int64)&v55, (__int64)v15, v53, v51, v49, v57, v58, v47, 0, a3);
  v32 = *(_QWORD *)(a1 + 552);
  v33 = v30;
  v34 = v31;
  if ( v30 )
  {
    v54 = *(_QWORD *)(a1 + 552);
    nullsub_686();
    *(_QWORD *)(v54 + 176) = v33;
    *(_DWORD *)(v54 + 184) = v34;
    sub_1D23870();
  }
  else
  {
    *(_QWORD *)(v32 + 176) = 0;
    *(_DWORD *)(v32 + 184) = v31;
  }
  *(_QWORD *)&v60 = a2;
  result = sub_205F5C0(a1 + 8, (__int64 *)&v60);
  v36 = v55;
  result[1] = v33;
  *((_DWORD *)result + 4) = v34;
  if ( v36 )
    return (__int64 *)sub_161E7C0((__int64)&v55, v36);
  return result;
}
