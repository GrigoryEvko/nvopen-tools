// Function: sub_17C2750
// Address: 0x17c2750
//
__int64 __fastcall sub_17C2750(__int64 a1, __int64 a2, unsigned __int64 a3, unsigned __int64 a4, char a5, __int64 *a6)
{
  unsigned __int64 v7; // r12
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // r14
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rdi
  __int64 v13; // rdx
  unsigned __int8 v14; // al
  __int64 v15; // rax
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 v18; // r15
  __int64 v19; // r12
  __int64 v20; // rbx
  char *v21; // rbx
  char *v22; // r12
  char *v23; // rdi
  _QWORD *v24; // rbx
  _QWORD *v25; // r12
  _QWORD *v26; // rdi
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  unsigned __int64 v32; // [rsp+10h] [rbp-530h]
  __int64 v34; // [rsp+28h] [rbp-518h] BYREF
  _QWORD v35[2]; // [rsp+30h] [rbp-510h] BYREF
  __int64 v36; // [rsp+40h] [rbp-500h] BYREF
  __int64 *v37; // [rsp+50h] [rbp-4F0h]
  __int64 v38; // [rsp+60h] [rbp-4E0h] BYREF
  _QWORD v39[2]; // [rsp+90h] [rbp-4B0h] BYREF
  __int64 v40; // [rsp+A0h] [rbp-4A0h] BYREF
  __int64 *v41; // [rsp+B0h] [rbp-490h]
  __int64 v42; // [rsp+C0h] [rbp-480h] BYREF
  _QWORD v43[2]; // [rsp+F0h] [rbp-450h] BYREF
  __int64 v44; // [rsp+100h] [rbp-440h] BYREF
  __int64 *v45; // [rsp+110h] [rbp-430h]
  __int64 v46; // [rsp+120h] [rbp-420h] BYREF
  void *v47; // [rsp+150h] [rbp-3F0h] BYREF
  int v48; // [rsp+158h] [rbp-3E8h]
  char v49; // [rsp+15Ch] [rbp-3E4h]
  __int64 v50; // [rsp+160h] [rbp-3E0h]
  __m128i v51; // [rsp+168h] [rbp-3D8h]
  __int64 v52; // [rsp+178h] [rbp-3C8h]
  __int64 v53; // [rsp+180h] [rbp-3C0h]
  __m128i v54; // [rsp+188h] [rbp-3B8h]
  __int64 v55; // [rsp+198h] [rbp-3A8h]
  _BYTE *v57; // [rsp+1A8h] [rbp-398h] BYREF
  __int64 v58; // [rsp+1B0h] [rbp-390h]
  _BYTE v59[356]; // [rsp+1B8h] [rbp-388h] BYREF
  int v60; // [rsp+31Ch] [rbp-224h]
  __int64 v61; // [rsp+320h] [rbp-220h]
  unsigned int *v62; // [rsp+330h] [rbp-210h] BYREF
  __int64 v63; // [rsp+338h] [rbp-208h]
  _DWORD v64[18]; // [rsp+340h] [rbp-200h] BYREF
  char *v65; // [rsp+388h] [rbp-1B8h]
  unsigned int v66; // [rsp+390h] [rbp-1B0h]
  char v67; // [rsp+398h] [rbp-1A8h] BYREF

  v7 = a3;
  v9 = a4 - a3;
  v10 = v9;
  if ( v9 >= a3 )
    a3 = v9;
  v11 = 1;
  if ( a3 > 0xFFFFFFFE )
    v11 = a3 / 0xFFFFFFFF + 1;
  v32 = v11;
  v34 = sub_16498A0(a1);
  v12 = 0;
  v13 = sub_161BE60(&v34, v7 / v32, v10 / v32);
  v14 = *(_BYTE *)(a1 + 16);
  if ( v14 > 0x17u )
  {
    if ( v14 == 78 )
    {
      v12 = a1 | 4;
    }
    else if ( v14 == 29 )
    {
      v12 = a1 & 0xFFFFFFFFFFFFFFFBLL;
    }
  }
  v15 = sub_1AB2980(v12, a2, v13);
  v16 = v15;
  if ( a5 )
  {
    v64[0] = v7;
    v62 = v64;
    v63 = 0x100000001LL;
    v47 = (void *)sub_16498A0(v15);
    v28 = sub_161BD30(&v47, v62, (unsigned int)v63);
    sub_1625C10(v16, 2, v28);
    if ( v62 != v64 )
      _libc_free((unsigned __int64)v62);
  }
  if ( a6 )
  {
    v17 = sub_15E0530(*a6);
    if ( sub_1602790(v17)
      || (v29 = sub_15E0530(*a6),
          v30 = sub_16033E0(v29),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v30 + 48LL))(v30)) )
    {
      sub_15CA3B0((__int64)&v62, (__int64)"pgo-icall-prom", (__int64)"Promoted", 8, a1);
      sub_15CAB20((__int64)&v62, "Promote indirect call to ", 0x19u);
      sub_15C9340((__int64)v43, "DirectCallee", 0xCu, a2);
      v18 = sub_17C2270((__int64)&v62, (__int64)v43);
      sub_15CAB20(v18, " with count ", 0xCu);
      sub_15C9D40((__int64)v39, "Count", 5, v7);
      v19 = sub_17C2270(v18, (__int64)v39);
      sub_15CAB20(v19, " out of ", 8u);
      sub_15C9D40((__int64)v35, "TotalCount", 10, a4);
      v20 = sub_17C2270(v19, (__int64)v35);
      v48 = *(_DWORD *)(v20 + 8);
      v49 = *(_BYTE *)(v20 + 12);
      v50 = *(_QWORD *)(v20 + 16);
      v51 = _mm_loadu_si128((const __m128i *)(v20 + 24));
      v52 = *(_QWORD *)(v20 + 40);
      v47 = &unk_49ECF68;
      v53 = *(_QWORD *)(v20 + 48);
      v54 = _mm_loadu_si128((const __m128i *)(v20 + 56));
      if ( *(_BYTE *)(v20 + 80) )
        v55 = *(_QWORD *)(v20 + 72);
      v57 = v59;
      v58 = 0x400000000LL;
      if ( *(_DWORD *)(v20 + 96) )
        sub_17C24C0((__int64)&v57, v20 + 88);
      v59[352] = *(_BYTE *)(v20 + 456);
      v60 = *(_DWORD *)(v20 + 460);
      v61 = *(_QWORD *)(v20 + 464);
      v47 = &unk_49ECF98;
      if ( v37 != &v38 )
        j_j___libc_free_0(v37, v38 + 1);
      if ( (__int64 *)v35[0] != &v36 )
        j_j___libc_free_0(v35[0], v36 + 1);
      if ( v41 != &v42 )
        j_j___libc_free_0(v41, v42 + 1);
      if ( (__int64 *)v39[0] != &v40 )
        j_j___libc_free_0(v39[0], v40 + 1);
      if ( v45 != &v46 )
        j_j___libc_free_0(v45, v46 + 1);
      if ( (__int64 *)v43[0] != &v44 )
        j_j___libc_free_0(v43[0], v44 + 1);
      v21 = v65;
      v62 = (unsigned int *)&unk_49ECF68;
      v22 = &v65[88 * v66];
      if ( v65 != v22 )
      {
        do
        {
          v22 -= 88;
          v23 = (char *)*((_QWORD *)v22 + 4);
          if ( v23 != v22 + 48 )
            j_j___libc_free_0(v23, *((_QWORD *)v22 + 6) + 1LL);
          if ( *(char **)v22 != v22 + 16 )
            j_j___libc_free_0(*(_QWORD *)v22, *((_QWORD *)v22 + 2) + 1LL);
        }
        while ( v21 != v22 );
        v22 = v65;
      }
      if ( v22 != &v67 )
        _libc_free((unsigned __int64)v22);
      sub_143AA50(a6, (__int64)&v47);
      v24 = v57;
      v47 = &unk_49ECF68;
      v25 = &v57[88 * (unsigned int)v58];
      if ( v57 != (_BYTE *)v25 )
      {
        do
        {
          v25 -= 11;
          v26 = (_QWORD *)v25[4];
          if ( v26 != v25 + 6 )
            j_j___libc_free_0(v26, v25[6] + 1LL);
          if ( (_QWORD *)*v25 != v25 + 2 )
            j_j___libc_free_0(*v25, v25[2] + 1LL);
        }
        while ( v24 != v25 );
        v25 = v57;
      }
      if ( v25 != (_QWORD *)v59 )
        _libc_free((unsigned __int64)v25);
    }
  }
  return v16;
}
