// Function: sub_1AC3A00
// Address: 0x1ac3a00
//
__int64 __fastcall sub_1AC3A00(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rax
  _QWORD *v14; // r15
  __int64 v15; // rdx
  _QWORD *v16; // rax
  __int64 v17; // r13
  __int64 v18; // rdx
  __int64 v19; // r15
  unsigned __int64 v20; // rbx
  _QWORD *v21; // rdi
  __int64 v22; // r13
  int v23; // ecx
  unsigned int v24; // ecx
  __int64 v25; // rdx
  _QWORD *v26; // rax
  double v27; // xmm4_8
  double v28; // xmm5_8
  __int64 v29; // r15
  _QWORD *v31; // r13
  unsigned int v32; // r15d
  __int64 v33; // r12
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // r9
  _QWORD *v37; // rax
  unsigned int v38; // [rsp+8h] [rbp-68h]
  __int64 v39; // [rsp+8h] [rbp-68h]
  __int64 v40; // [rsp+8h] [rbp-68h]
  const char *v41; // [rsp+10h] [rbp-60h] BYREF
  __int64 v42; // [rsp+18h] [rbp-58h]
  const char **v43; // [rsp+20h] [rbp-50h] BYREF
  char *v44; // [rsp+28h] [rbp-48h]
  __int16 v45; // [rsp+30h] [rbp-40h]

  v11 = *(_QWORD *)(a1 + 8);
  if ( v11 )
  {
    v12 = sub_15F2050(a1);
    v13 = sub_1632FA0(v12);
    if ( a2 )
    {
      v14 = *(_QWORD **)a1;
      v38 = *(_DWORD *)(v13 + 4);
      v41 = sub_1649960(a1);
      v43 = &v41;
      v42 = v15;
      v45 = 773;
      v44 = ".reg2mem";
      v16 = sub_1648A60(64, 1u);
      v11 = (__int64)v16;
      if ( v16 )
        sub_15F8BC0((__int64)v16, v14, v38, 0, (__int64)&v43, a2);
    }
    else
    {
      v31 = *(_QWORD **)a1;
      v32 = *(_DWORD *)(v13 + 4);
      v33 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 56LL);
      v41 = sub_1649960(a1);
      v42 = v34;
      v43 = &v41;
      v45 = 773;
      v44 = ".reg2mem";
      v35 = *(_QWORD *)(v33 + 80);
      if ( !v35 )
        BUG();
      v36 = *(_QWORD *)(v35 + 24);
      if ( v36 )
        v36 -= 24;
      v40 = v36;
      v37 = sub_1648A60(64, 1u);
      v11 = (__int64)v37;
      if ( v37 )
        sub_15F8BC0((__int64)v37, v31, v32, 0, (__int64)&v43, v40);
    }
    v17 = 0;
    v39 = 8LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
    if ( (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) != 0 )
    {
      do
      {
        if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
          v18 = *(_QWORD *)(a1 - 8);
        else
          v18 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
        v19 = *(_QWORD *)(v18 + 3 * v17);
        v20 = sub_157EBA0(*(_QWORD *)(v17 + v18 + 24LL * *(unsigned int *)(a1 + 56) + 8));
        v21 = sub_1648A60(64, 2u);
        if ( v21 )
          sub_15F9660((__int64)v21, v19, v11, v20);
        v17 += 8;
      }
      while ( v39 != v17 );
    }
    v22 = a1 + 24;
    while ( 1 )
    {
      v23 = *(unsigned __int8 *)(v22 - 8);
      if ( (_BYTE)v23 != 77 )
      {
        v24 = v23 - 34;
        if ( v24 > 0x36 || ((1LL << v24) & 0x40018000000001LL) == 0 )
          break;
      }
      v22 = *(_QWORD *)(v22 + 8);
      if ( !v22 )
        BUG();
    }
    v41 = sub_1649960(a1);
    v45 = 773;
    v42 = v25;
    v43 = &v41;
    v44 = ".reload";
    v26 = sub_1648A60(64, 1u);
    v29 = (__int64)v26;
    if ( v26 )
      sub_15F90E0((__int64)v26, v11, (__int64)&v43, v22 - 24);
    sub_164D160(a1, v29, a3, a4, a5, a6, v27, v28, a9, a10);
  }
  sub_15F20C0((_QWORD *)a1);
  return v11;
}
