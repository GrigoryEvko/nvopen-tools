// Function: sub_3757FE0
// Address: 0x3757fe0
//
__int64 __fastcall sub_3757FE0(__int64 *a1, unsigned __int8 *a2, __m128i *a3, unsigned __int8 a4, unsigned __int8 a5)
{
  __int64 v7; // rdx
  _QWORD *v8; // rax
  __int64 v9; // rdi
  __int64 v10; // r13
  _QWORD *v11; // rax
  __int64 v12; // r8
  __int64 v13; // r9
  unsigned __int8 *v14; // rsi
  unsigned __int8 *v15; // r14
  _QWORD *v16; // rax
  __int64 v17; // rax
  int v18; // edi
  __int64 v19; // rdx
  __int64 v20; // rcx
  unsigned __int64 v21; // r10
  __int64 v22; // r13
  unsigned __int64 v23; // r11
  __int64 v24; // rdx
  unsigned __int64 *v25; // rax
  unsigned __int64 *v26; // rdx
  __int64 v27; // rax
  _QWORD *v28; // r13
  int v29; // eax
  __int64 v30; // rax
  _QWORD *v31; // r13
  unsigned int v32; // r15d
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // r13
  __int64 *v36; // r14
  __int64 v37; // rdx
  __int64 v38; // rax
  unsigned __int64 v40; // [rsp+10h] [rbp-E0h]
  int v41; // [rsp+18h] [rbp-D8h]
  int v42; // [rsp+24h] [rbp-CCh]
  _QWORD *v43; // [rsp+28h] [rbp-C8h]
  _QWORD *v46; // [rsp+38h] [rbp-B8h]
  _QWORD *v47; // [rsp+38h] [rbp-B8h]
  _QWORD *v48; // [rsp+38h] [rbp-B8h]
  int v49; // [rsp+38h] [rbp-B8h]
  __int64 v50; // [rsp+40h] [rbp-B0h]
  unsigned __int8 *v52; // [rsp+58h] [rbp-98h] BYREF
  unsigned __int8 *v53; // [rsp+60h] [rbp-90h] BYREF
  __int64 v54; // [rsp+68h] [rbp-88h]
  unsigned __int8 *v55; // [rsp+70h] [rbp-80h] BYREF
  __int64 v56; // [rsp+78h] [rbp-78h]
  __int64 v57; // [rsp+80h] [rbp-70h] BYREF
  __m128i v58; // [rsp+90h] [rbp-60h] BYREF
  __int64 v59; // [rsp+A0h] [rbp-50h]
  __int64 v60; // [rsp+A8h] [rbp-48h]
  __int64 v61; // [rsp+B0h] [rbp-40h]

  v7 = *(_QWORD *)(**((_QWORD **)a2 + 5) + 96LL);
  v8 = *(_QWORD **)(v7 + 24);
  if ( *(_DWORD *)(v7 + 32) > 0x40u )
    v8 = (_QWORD *)*v8;
  v9 = a1[3];
  v10 = a1[1];
  v43 = *(_QWORD **)(*(_QWORD *)(v9 + 280) + 8LL * (unsigned int)v8);
  v11 = sub_2FF6410(v9, v43);
  v42 = sub_2EC06C0(v10, (__int64)v11, byte_3F871B3, 0, v12, v13);
  v50 = *(_QWORD *)(a1[2] + 8) - 760LL;
  v14 = (unsigned __int8 *)*((_QWORD *)a2 + 10);
  v52 = v14;
  if ( !v14 )
  {
    v55 = 0;
    goto LABEL_34;
  }
  sub_B96E90((__int64)&v52, (__int64)v14, 1);
  v55 = v52;
  if ( !v52 )
  {
LABEL_34:
    v15 = (unsigned __int8 *)*a1;
    v56 = 0;
    v57 = 0;
    v53 = 0;
    goto LABEL_7;
  }
  sub_B976B0((__int64)&v52, v52, (__int64)&v55);
  v56 = 0;
  v52 = 0;
  v15 = (unsigned __int8 *)*a1;
  v57 = 0;
  v53 = v55;
  if ( v55 )
    sub_B96E90((__int64)&v53, (__int64)v55, 1);
LABEL_7:
  v16 = sub_2E7B380(v15, v50, &v53, 0);
  if ( v56 )
  {
    v46 = v16;
    sub_2E882B0((__int64)v16, (__int64)v15, v56);
    v16 = v46;
  }
  if ( v57 )
  {
    v47 = v16;
    sub_2E88680((__int64)v16, (__int64)v15, v57);
    v16 = v47;
  }
  v58.m128i_i64[0] = 0x10000000;
  v48 = v16;
  v58.m128i_i32[2] = v42;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  sub_2E8EAD0((__int64)v16, (__int64)v15, &v58);
  v17 = (__int64)v48;
  if ( v53 )
  {
    sub_B91220((__int64)&v53, (__int64)v53);
    v17 = (__int64)v48;
  }
  v53 = v15;
  v54 = v17;
  if ( v55 )
    sub_B91220((__int64)&v55, (__int64)v55);
  if ( v52 )
    sub_B91220((__int64)&v52, (__int64)v52);
  v18 = *((_DWORD *)a2 + 16);
  v19 = *((_QWORD *)a2 + 5);
  v49 = v18;
  if ( !v18 )
    goto LABEL_21;
  v20 = (unsigned int)(v18 - 1);
  if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)(v19 + 40 * v20) + 48LL) + 16LL * *(unsigned int *)(v19 + 40 * v20 + 8)) != 1 )
    LODWORD(v20) = *((_DWORD *)a2 + 16);
  v49 = v20;
  if ( (_DWORD)v20 != 1 )
  {
LABEL_21:
    v21 = *(_QWORD *)(v19 + 40);
    v22 = 1;
    v23 = *(_QWORD *)(v19 + 48);
    while ( 1 )
    {
      v32 = v22 + 1;
      sub_3752760(a1, (__int64 *)&v53, v21, v23, v22 + 1, v50, (__int64)a3, 0, a4, a5);
      if ( (_DWORD)v22 + 1 == v49 )
        break;
      v24 = *((_QWORD *)a2 + 5);
      v25 = (unsigned __int64 *)(v24 + 40LL * v32);
      v21 = *v25;
      v23 = v25[1];
      if ( (v32 & 1) == 0 )
      {
        v26 = (unsigned __int64 *)(v24 + 40 * v22);
        if ( *(_DWORD *)(*v26 + 24) != 9 || (unsigned int)(*(_DWORD *)(*v26 + 96) - 1) > 0x3FFFFFFE )
        {
          v27 = *(_QWORD *)(*v25 + 96);
          v28 = *(_QWORD **)(v27 + 24);
          if ( *(_DWORD *)(v27 + 32) > 0x40u )
            v28 = (_QWORD *)*v28;
          v40 = v21;
          v41 = v23;
          v29 = sub_3752000(a1, *v26, v26[1], (__int64)a3, v33, v34);
          v30 = (*(__int64 (__fastcall **)(__int64, _QWORD *, unsigned __int64, _QWORD))(*(_QWORD *)a1[3] + 256LL))(
                  a1[3],
                  v43,
                  *(_QWORD *)(*(_QWORD *)(a1[1] + 56) + 16LL * (v29 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
                  (unsigned int)v28);
          v21 = v40;
          LODWORD(v23) = v41;
          v31 = (_QWORD *)v30;
          if ( v30 && v43 != (_QWORD *)v30 )
          {
            sub_2EBE4E0(a1[1], v42, v30);
            v43 = v31;
            v21 = v40;
            LODWORD(v23) = v41;
          }
        }
      }
      v22 = v32;
    }
  }
  v35 = v54;
  v36 = (__int64 *)a1[6];
  sub_2E31040((__int64 *)(a1[5] + 40), v54);
  v37 = *v36;
  v38 = *(_QWORD *)v35;
  *(_QWORD *)(v35 + 8) = v36;
  v37 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v35 = v37 | v38 & 7;
  *(_QWORD *)(v37 + 8) = v35;
  *v36 = *v36 & 7 | v35;
  v55 = a2;
  LODWORD(v56) = 0;
  LODWORD(v57) = v42;
  return sub_3755010((__int64)&v58, a3, (unsigned __int64 *)&v55, &v57);
}
