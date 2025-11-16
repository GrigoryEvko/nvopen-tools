// Function: sub_1BAD600
// Address: 0x1bad600
//
__int64 __fastcall sub_1BAD600(
        __int64 a1,
        __int64 *a2,
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
  _QWORD *v12; // r12
  unsigned __int64 v13; // rax
  __int64 result; // rax
  __int64 v15; // rdx
  __int64 v16; // r13
  unsigned __int64 v17; // rax
  __int64 v18; // r15
  _QWORD *v19; // rax
  _QWORD *v20; // r14
  unsigned __int64 v21; // rax
  double v22; // xmm4_8
  double v23; // xmm5_8
  int v24; // r8d
  int v25; // r9d
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // r8
  __int64 v29; // r14
  __int64 v30; // r15
  __int64 *v31; // r13
  __int64 v32; // rax
  __int64 v33; // r12
  __int64 v34; // rdi
  __int64 v35; // [rsp+0h] [rbp-60h]
  __int64 v37; // [rsp+8h] [rbp-58h]
  __int64 v38[2]; // [rsp+10h] [rbp-50h] BYREF
  char v39; // [rsp+20h] [rbp-40h]
  char v40; // [rsp+21h] [rbp-3Fh]

  v12 = (_QWORD *)sub_13FC520((__int64)a2);
  v13 = sub_157EBA0((__int64)v12);
  result = sub_3862440(*(_QWORD *)(*(_QWORD *)(a1 + 448) + 48LL), v13);
  if ( v15 )
  {
    v16 = v15;
    v40 = 1;
    v39 = 3;
    v38[0] = (__int64)"vector.memcheck";
    sub_164B780((__int64)v12, v38);
    v40 = 1;
    v38[0] = (__int64)"vector.ph";
    v39 = 3;
    v17 = sub_157EBA0((__int64)v12);
    v18 = sub_157FBF0(v12, (__int64 *)(v17 + 24), (__int64)v38);
    sub_1BACEB0(*(_QWORD *)(a1 + 32), v18, (__int64)v12);
    if ( *a2 )
      sub_1400330(*a2, v18, *(_QWORD *)(a1 + 24));
    v19 = sub_1648A60(56, 3u);
    v20 = v19;
    if ( v19 )
      sub_15F83E0((__int64)v19, a3, v18, v16, 0);
    v21 = sub_157EBA0((__int64)v12);
    sub_1AA6530(v21, v20, a4, a5, a6, a7, v22, v23, a10, a11);
    v26 = *(unsigned int *)(a1 + 224);
    if ( (unsigned int)v26 >= *(_DWORD *)(a1 + 228) )
    {
      sub_16CD150(a1 + 216, (const void *)(a1 + 232), 0, 8, v24, v25);
      v26 = *(unsigned int *)(a1 + 224);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 216) + 8 * v26) = v12;
    v27 = *(_QWORD *)(a1 + 16);
    v28 = *(_QWORD *)(a1 + 32);
    v29 = *(_QWORD *)(a1 + 8);
    *(_BYTE *)(a1 + 464) = 1;
    ++*(_DWORD *)(a1 + 224);
    v30 = *(_QWORD *)(a1 + 24);
    v37 = v28;
    v35 = *(_QWORD *)(v27 + 112);
    v31 = *(__int64 **)(*(_QWORD *)(a1 + 448) + 48LL);
    v32 = sub_22077B0(520);
    v33 = v32;
    if ( v32 )
      sub_1B1E040(v32, v31, v29, v30, v37, v35, 1);
    v34 = *(_QWORD *)(a1 + 80);
    *(_QWORD *)(a1 + 80) = v33;
    if ( v34 )
    {
      sub_1B90BF0(v34);
      v33 = *(_QWORD *)(a1 + 80);
    }
    return sub_1B205A0(v33);
  }
  return result;
}
