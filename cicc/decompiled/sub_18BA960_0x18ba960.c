// Function: sub_18BA960
// Address: 0x18ba960
//
__int64 __fastcall sub_18BA960(
        __int64 *a1,
        __int64 *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // rax
  unsigned int v12; // eax
  __int64 v13; // rsi
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // rcx
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rsi
  unsigned __int64 v19; // rcx
  unsigned __int64 v20; // rax
  char *v21; // r12
  size_t v22; // r14
  __int64 v23; // rax
  char *v24; // rdx
  char *v25; // r12
  char v26; // si
  __int64 *v27; // rax
  __int64 *v28; // rax
  __int64 v29; // rax
  char *v30; // r14
  __int64 v31; // r12
  size_t v32; // r12
  __int64 *v33; // rax
  __int64 *v34; // rax
  __int64 **v35; // rax
  __int64 v36; // rcx
  __int64 *v37; // r15
  char v38; // cl
  _QWORD *v39; // rax
  __int64 v40; // r12
  size_t v41; // rdx
  const void *v42; // rsi
  __int64 *v43; // rax
  __int64 v44; // rdi
  __int64 v45; // rdi
  __int64 v46; // rax
  __int64 v47; // rcx
  __int64 v48; // r12
  char v49; // al
  double v50; // xmm4_8
  double v51; // xmm5_8
  __int64 result; // rax
  __int64 v53; // rax
  __int64 v54; // [rsp+0h] [rbp-70h]
  char v55; // [rsp+8h] [rbp-68h]
  __int64 v56; // [rsp+8h] [rbp-68h]
  __int64 *v57[2]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v58[2]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v59; // [rsp+30h] [rbp-40h]

  if ( a2[2] != a2[3] || (result = a2[9], a2[8] != result) )
  {
    v11 = sub_1632FA0(*a1);
    v12 = sub_15A9520(v11, 0);
    v13 = a2[2];
    v14 = v12;
    v15 = a2[3] - v13;
    v16 = v12 * ((v12 + v15 - 1) / v12);
    if ( v16 > v15 )
    {
      sub_CD93F0(a2 + 2, v16 - v15);
    }
    else if ( v16 < v15 )
    {
      v17 = v13 + v16;
      if ( a2[3] != v17 )
        a2[3] = v17;
    }
    v18 = a2[8];
    v19 = a2[9] - v18;
    v20 = v14 * ((v14 + v19 - 1) / v14);
    if ( v19 < v20 )
    {
      sub_CD93F0(a2 + 8, v20 - v19);
    }
    else if ( v19 > v20 )
    {
      v53 = v18 + v20;
      if ( a2[9] != v53 )
        a2[9] = v53;
    }
    v21 = (char *)a2[2];
    v22 = a2[3] - (_QWORD)v21;
    if ( v22 >> 1 )
    {
      v23 = 0;
      do
      {
        v24 = &v21[v22 - 1 - v23];
        v25 = &v21[v23++];
        v26 = *v25;
        *v25 = *v24;
        *v24 = v26;
        v21 = (char *)a2[2];
      }
      while ( v23 != v22 >> 1 );
      v22 = a2[3] - (_QWORD)v21;
    }
    v27 = (__int64 *)sub_1644C60(*(_QWORD **)*a1, 8u);
    v28 = sub_1645D80(v27, v22);
    v29 = sub_15991C0(v21, v22, (__int64 **)v28);
    v30 = (char *)a2[8];
    v31 = a2[9];
    v58[0] = v29;
    v32 = v31 - (_QWORD)v30;
    v58[1] = *(_QWORD *)(*a2 - 24);
    v33 = (__int64 *)sub_1644C60(*(_QWORD **)*a1, 8u);
    v34 = sub_1645D80(v33, v32);
    v59 = sub_15991C0(v30, v32, (__int64 **)v34);
    v35 = (__int64 **)sub_15943F0(v58, 3, 0);
    v37 = (__int64 *)sub_159F090(v35, v58, 3, v36);
    v38 = *(_BYTE *)(*a2 + 80);
    v54 = *v37;
    LOWORD(v59) = 257;
    v55 = v38 & 1;
    v39 = sub_1648A60(88, 1u);
    v40 = (__int64)v39;
    if ( v39 )
      sub_15E51E0((__int64)v39, *a1, v54, v55, 8, (__int64)v37, (__int64)v58, *a2, 0, 0, 0);
    v41 = 0;
    v42 = 0;
    if ( (*(_BYTE *)(*a2 + 34) & 0x20) != 0 )
      v42 = (const void *)sub_15E61A0(*a2);
    sub_15E5D20(v40, v42, v41);
    *(_QWORD *)(v40 + 48) = *(_QWORD *)(*a2 + 48);
    sub_1628980(v40, *a2, *((_DWORD *)a2 + 6) - *((_DWORD *)a2 + 4));
    v56 = *a1;
    v43 = (__int64 *)sub_159C470(a1[7], 0, 0);
    v44 = a1[7];
    v57[0] = v43;
    v57[1] = (__int64 *)sub_159C470(v44, 1, 0);
    v45 = *v37;
    BYTE4(v58[0]) = 0;
    v46 = sub_15A2E80(v45, v40, v57, 2u, 0, (__int64)v58, 0);
    v47 = *a2;
    LOWORD(v59) = 257;
    v48 = sub_15E57E0(**(_QWORD **)(v47 - 24), 0, *(_BYTE *)(v47 + 32) & 0xF, (__int64)v58, v46, v56);
    v49 = *(_BYTE *)(*a2 + 32) & 0x30 | *(_BYTE *)(v48 + 32) & 0xCF;
    *(_BYTE *)(v48 + 32) = v49;
    if ( (v49 & 0xFu) - 7 <= 1 || (v49 & 0x30) != 0 && (v49 & 0xF) != 9 )
      *(_BYTE *)(v48 + 33) |= 0x40u;
    sub_164B7C0(v48, *a2);
    sub_164D160(*a2, v48, a3, a4, a5, a6, v50, v51, a9, a10);
    return sub_15E55B0(*a2);
  }
  return result;
}
