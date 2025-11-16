// Function: sub_2D423B0
// Address: 0x2d423b0
//
__int64 __fastcall sub_2D423B0(__int64 a1, __m128i a2, __m128i a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // rcx
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // r8
  int v16; // eax
  _QWORD *v17; // rbx
  unsigned __int64 v18; // r13
  __int64 v19; // r15
  unsigned __int64 v20; // r12
  __int64 v21; // rsi
  char *v22; // [rsp+20h] [rbp-170h] BYREF
  __int64 v23; // [rsp+28h] [rbp-168h]
  _BYTE v24[40]; // [rsp+30h] [rbp-160h] BYREF
  char *v25; // [rsp+58h] [rbp-138h] BYREF
  __int64 v26; // [rsp+60h] [rbp-130h]
  char v27; // [rsp+68h] [rbp-128h] BYREF
  int v28; // [rsp+88h] [rbp-108h]
  __int64 v29; // [rsp+90h] [rbp-100h]
  __int64 v30; // [rsp+98h] [rbp-F8h]
  __int64 v31; // [rsp+A0h] [rbp-F0h]
  int v32; // [rsp+A8h] [rbp-E8h]
  __int64 v33; // [rsp+B0h] [rbp-E0h] BYREF
  int v34; // [rsp+B8h] [rbp-D8h] BYREF
  unsigned __int64 v35; // [rsp+C0h] [rbp-D0h]
  int *v36; // [rsp+C8h] [rbp-C8h]
  int *v37; // [rsp+D0h] [rbp-C0h]
  __int64 v38; // [rsp+D8h] [rbp-B8h]
  unsigned __int64 v39; // [rsp+E0h] [rbp-B0h]
  __int64 v40; // [rsp+E8h] [rbp-A8h]
  __int64 v41; // [rsp+F0h] [rbp-A0h]
  void *s; // [rsp+F8h] [rbp-98h]
  __int64 v43; // [rsp+100h] [rbp-90h]
  _QWORD *v44; // [rsp+108h] [rbp-88h]
  __int64 v45; // [rsp+110h] [rbp-80h]
  int v46; // [rsp+118h] [rbp-78h]
  __int64 v47; // [rsp+120h] [rbp-70h]
  __int64 v48; // [rsp+128h] [rbp-68h] BYREF
  _QWORD v49[2]; // [rsp+130h] [rbp-60h] BYREF
  char v50; // [rsp+140h] [rbp-50h] BYREF

  v5 = a1 + 72;
  if ( (unsigned __int8)sub_AEA460(*(_QWORD *)(a5 + 40)) )
  {
    v8 = sub_B2BEC0(a5);
    v34 = 0;
    v36 = &v34;
    v37 = &v34;
    s = &v48;
    v49[0] = &v50;
    v35 = 0;
    v38 = 0;
    v39 = 0;
    v40 = 0;
    v41 = 0;
    v43 = 1;
    v44 = 0;
    v45 = 0;
    v46 = 1065353216;
    v47 = 0;
    v48 = 0;
    v49[1] = 0x100000000LL;
    sub_2D3F710(a5, v8, &v33, a2, a3);
    v22 = v24;
    v23 = 0x100000000LL;
    v25 = &v27;
    v26 = 0x100000000LL;
    v28 = 0;
    v29 = 0;
    v30 = 0;
    v31 = 0;
    v32 = 0;
    sub_2D2C8E0((__int64)&v22, (__int64)&v33, v9, v10, v11, (__int64)&v33);
    v14 = (unsigned int)v23;
    *(_QWORD *)(a1 + 8) = 0x100000000LL;
    v15 = a1 + 16;
    *(_QWORD *)a1 = a1 + 16;
    if ( (_DWORD)v14 )
      sub_2D23140(a1, &v22, v14, v12, v15, v13);
    *(_QWORD *)(a1 + 56) = v5;
    *(_QWORD *)(a1 + 64) = 0x100000000LL;
    if ( (_DWORD)v26 )
      sub_2D29780((unsigned int *)(a1 + 56), (__int64)&v25, v14, v12, v15, v13);
    v16 = v28;
    *(_QWORD *)(a1 + 112) = 1;
    *(_DWORD *)(a1 + 104) = v16;
    ++v29;
    *(_QWORD *)(a1 + 120) = v30;
    v30 = 0;
    *(_QWORD *)(a1 + 128) = v31;
    v31 = 0;
    *(_DWORD *)(a1 + 136) = v32;
    v32 = 0;
    sub_C7D6A0(0, 0, 8);
    sub_2D288B0((__int64)&v25);
    if ( v22 != v24 )
      _libc_free((unsigned __int64)v22);
    sub_2D288B0((__int64)v49);
    v17 = v44;
    while ( v17 )
    {
      v18 = (unsigned __int64)v17;
      v17 = (_QWORD *)*v17;
      v19 = *(_QWORD *)(v18 + 16);
      v20 = v19 + 32LL * *(unsigned int *)(v18 + 24);
      if ( v19 != v20 )
      {
        do
        {
          v21 = *(_QWORD *)(v20 - 16);
          v20 -= 32LL;
          if ( v21 )
            sub_B91220(v20 + 16, v21);
        }
        while ( v19 != v20 );
        v20 = *(_QWORD *)(v18 + 16);
      }
      if ( v20 != v18 + 32 )
        _libc_free(v20);
      j_j___libc_free_0(v18);
    }
    memset(s, 0, 8 * v43);
    v45 = 0;
    v44 = 0;
    if ( s != &v48 )
      j_j___libc_free_0((unsigned __int64)s);
    if ( v39 )
      j_j___libc_free_0(v39);
    sub_2D24760(v35);
  }
  else
  {
    memset((void *)a1, 0, 0x90u);
    *(_QWORD *)a1 = a1 + 16;
    *(_DWORD *)(a1 + 12) = 1;
    *(_QWORD *)(a1 + 56) = v5;
    *(_DWORD *)(a1 + 68) = 1;
  }
  return a1;
}
