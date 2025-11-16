// Function: sub_23A4250
// Address: 0x23a4250
//
void __fastcall sub_23A4250(__int64 a1, unsigned __int64 *a2, unsigned __int64 a3, int a4, __int64 a5, int a6)
{
  int v6; // eax
  char v7; // r12
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  _QWORD *v14; // rbx
  _QWORD *v15; // r12
  __int64 v17; // [rsp+70h] [rbp-AB0h] BYREF
  __int64 v18; // [rsp+78h] [rbp-AA8h]
  __int64 v19; // [rsp+80h] [rbp-AA0h]
  unsigned __int64 v20[6]; // [rsp+90h] [rbp-A90h] BYREF
  _DWORD v21[20]; // [rsp+C0h] [rbp-A60h] BYREF
  int v22; // [rsp+110h] [rbp-A10h]
  _BYTE v23[104]; // [rsp+120h] [rbp-A00h] BYREF
  _QWORD *v24; // [rsp+188h] [rbp-998h] BYREF
  _QWORD *v25; // [rsp+190h] [rbp-990h]
  char v26[40]; // [rsp+1B0h] [rbp-970h] BYREF
  char v27[40]; // [rsp+1D8h] [rbp-948h] BYREF
  __int64 v28[4]; // [rsp+200h] [rbp-920h] BYREF
  int v29; // [rsp+220h] [rbp-900h]
  char v30; // [rsp+224h] [rbp-8FCh]
  char v31; // [rsp+228h] [rbp-8F8h] BYREF
  __int64 v32; // [rsp+328h] [rbp-7F8h]
  __int64 v33; // [rsp+330h] [rbp-7F0h]
  __int64 v34; // [rsp+338h] [rbp-7E8h]
  int v35; // [rsp+340h] [rbp-7E0h]
  _QWORD *v36; // [rsp+348h] [rbp-7D8h]
  __int64 v37; // [rsp+350h] [rbp-7D0h]
  __int64 v38; // [rsp+358h] [rbp-7C8h]
  __int64 v39; // [rsp+360h] [rbp-7C0h]
  int v40; // [rsp+368h] [rbp-7B8h]
  __int64 v41; // [rsp+370h] [rbp-7B0h]
  _QWORD v42[7]; // [rsp+378h] [rbp-7A8h] BYREF
  _QWORD v43[4]; // [rsp+3B0h] [rbp-770h] BYREF
  int v44; // [rsp+3D0h] [rbp-750h]
  __int64 v45; // [rsp+3D8h] [rbp-748h]
  char *v46; // [rsp+3E0h] [rbp-740h]
  __int64 v47; // [rsp+3E8h] [rbp-738h]
  int v48; // [rsp+3F0h] [rbp-730h]
  char v49; // [rsp+3F4h] [rbp-72Ch]
  char v50; // [rsp+3F8h] [rbp-728h] BYREF

  if ( !(_BYTE)qword_4FDD148 )
  {
    memset(&v21[2], 0, 0x48u);
    v6 = qword_4FDD068;
    v21[16] = 65792;
    v21[0] = qword_4FDD068;
    if ( !HIDWORD(a3) )
      v6 = 325;
    LOBYTE(v21[2]) = 1;
    v21[1] = v6;
    v22 = 0;
    sub_26124A0(
      (unsigned int)v23,
      1,
      a4,
      0,
      0,
      a6,
      *(_OWORD *)&_mm_loadu_si128((const __m128i *)v21),
      *(_OWORD *)&_mm_loadu_si128((const __m128i *)&v21[4]),
      *(_OWORD *)&_mm_loadu_si128((const __m128i *)&v21[8]),
      *(_OWORD *)&_mm_loadu_si128((const __m128i *)&v21[12]),
      *(_OWORD *)&_mm_loadu_si128((const __m128i *)&v21[16]),
      0);
    memset(v20, 0, 40);
    sub_291E720(&v17, 0);
    v7 = v17;
    v8 = sub_22077B0(0x10u);
    if ( v8 )
    {
      *(_BYTE *)(v8 + 8) = v7;
      *(_QWORD *)v8 = &unk_4A11C38;
    }
    v28[0] = v8;
    sub_23A1F40(v20, (unsigned __int64 *)v28);
    if ( v28[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v28[0] + 8LL))(v28[0]);
    v9 = sub_22077B0(0x10u);
    if ( v9 )
    {
      *(_BYTE *)(v9 + 8) = 0;
      *(_QWORD *)v9 = &unk_4A118B8;
    }
    v28[0] = v9;
    sub_23A1F40(v20, (unsigned __int64 *)v28);
    sub_233EFE0(v28);
    v17 = 0x100010000000001LL;
    v18 = 0x1000101000000LL;
    v19 = 0;
    sub_29744D0(v28, &v17);
    sub_23A1F80(v20, v28);
    LOBYTE(v17) = 0;
    HIDWORD(v17) = 1;
    LOBYTE(v18) = 0;
    sub_F10C20((__int64)v28, v17, v18);
    sub_2353C90(v20, (__int64)v28, v10, v11, v12, v13);
    sub_233BCC0((__int64)v28);
    sub_23A0D70(a1, (__int64)v20, a3);
    sub_234D2B0((__int64)v28, (__int64 *)v20, *(_BYTE *)(a1 + 32), 0);
    sub_235A8B0((unsigned __int64 *)&v24, v28);
    sub_233EFE0(v28);
    sub_2357600(a2, (__int64)v23);
    v28[2] = (__int64)&v31;
    v36 = v42;
    v42[1] = v43;
    v46 = &v50;
    LOBYTE(v28[0]) = 0;
    v28[1] = 0;
    v28[3] = 32;
    v29 = 0;
    v30 = 1;
    v32 = 0;
    v33 = 0;
    v34 = 0;
    v35 = 0;
    v37 = 1;
    v38 = 0;
    v39 = 0;
    v40 = 1065353216;
    v41 = 0;
    v42[0] = 0;
    v42[2] = 1;
    v42[3] = 0;
    v42[4] = 0;
    v42[5] = 1065353216;
    v42[6] = 0;
    memset(v43, 0, sizeof(v43));
    v44 = 0;
    v45 = 0;
    v47 = 32;
    v48 = 0;
    v49 = 1;
    sub_23A2670(a2, (__int64)v28);
    sub_233AAF0((__int64)v28);
    sub_233F7F0((__int64)v20);
    sub_234A900((__int64)v27);
    sub_234A900((__int64)v26);
    v14 = v25;
    v15 = v24;
    if ( v25 != v24 )
    {
      do
      {
        if ( *v15 )
          (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v15 + 8LL))(*v15);
        ++v15;
      }
      while ( v14 != v15 );
      v15 = v24;
    }
    if ( v15 )
      j_j___libc_free_0((unsigned __int64)v15);
  }
}
