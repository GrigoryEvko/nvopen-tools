// Function: sub_2F59FD0
// Address: 0x2f59fd0
//
void __fastcall sub_2F59FD0(_QWORD *a1)
{
  __int64 v2; // rax
  _QWORD **v3; // rbx
  _QWORD **i; // r13
  _QWORD *v5; // rdx
  __int64 v6; // r13
  __int64 v7; // rbx
  __int64 v8; // r13
  int v9; // esi
  unsigned int v10; // edx
  __int64 *v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rcx
  int v15; // eax
  __int64 **v16; // r13
  __int64 v17; // rax
  __int64 *v18; // rax
  __int64 v19; // rax
  __int64 v20; // rcx
  int v21; // esi
  __int64 v22; // rax
  __int64 *v23; // rdi
  _QWORD *v24; // rax
  __int64 v25; // r12
  __int64 v26; // rax
  unsigned __int64 *v27; // r13
  unsigned __int64 *v28; // r12
  unsigned __int64 v29; // rdi
  int v30; // eax
  int v31; // r8d
  __int64 v32; // rax
  __int64 v33; // rax
  unsigned __int8 *v34; // [rsp+8h] [rbp-228h] BYREF
  __m128i v35; // [rsp+10h] [rbp-220h] BYREF
  __int128 v36; // [rsp+20h] [rbp-210h] BYREF
  __int128 v37; // [rsp+30h] [rbp-200h]
  __int64 v38; // [rsp+40h] [rbp-1F0h]
  float v39; // [rsp+48h] [rbp-1E8h]
  unsigned __int8 *v40; // [rsp+50h] [rbp-1E0h] BYREF
  int v41; // [rsp+58h] [rbp-1D8h]
  int v42; // [rsp+5Ch] [rbp-1D4h]
  __int64 v43; // [rsp+60h] [rbp-1D0h]
  __m128i v44; // [rsp+68h] [rbp-1C8h]
  char *v45; // [rsp+78h] [rbp-1B8h]
  char *v46; // [rsp+80h] [rbp-1B0h]
  __int64 v47; // [rsp+88h] [rbp-1A8h]
  char v48; // [rsp+98h] [rbp-198h]
  unsigned __int64 *v49; // [rsp+A0h] [rbp-190h]
  __int64 v50; // [rsp+A8h] [rbp-188h]
  _BYTE v51[324]; // [rsp+B0h] [rbp-180h] BYREF
  int v52; // [rsp+1F4h] [rbp-3Ch]
  __int64 v53; // [rsp+1F8h] [rbp-38h]

  v2 = a1[101];
  v38 = 0;
  v39 = 0.0;
  v36 = 0;
  v37 = 0;
  v3 = *(_QWORD ***)(v2 + 40);
  for ( i = *(_QWORD ***)(v2 + 32); v3 != i; v39 = v39 + *(float *)&v45 )
  {
    v5 = *i++;
    sub_2F59B00((__int64)&v40, (__int64)a1, v5);
    LODWORD(v36) = (_DWORD)v40 + v36;
    DWORD1(v36) += HIDWORD(v40);
    *((float *)&v37 + 2) = *((float *)&v37 + 2) + *(float *)v44.m128i_i32;
    DWORD2(v36) += v41;
    HIDWORD(v36) += v42;
    *((float *)&v37 + 3) = *((float *)&v37 + 3) + *(float *)&v44.m128i_i32[1];
    LODWORD(v37) = v43 + v37;
    DWORD1(v37) += HIDWORD(v43);
    *(float *)&v38 = *(float *)&v38 + *(float *)&v44.m128i_i32[2];
    *((float *)&v38 + 1) = *((float *)&v38 + 1) + *(float *)&v44.m128i_i32[3];
  }
  v6 = a1[96];
  v7 = *(_QWORD *)(v6 + 328);
  v8 = v6 + 320;
  if ( v7 != v8 )
  {
    while ( 1 )
    {
      v13 = a1[101];
      v14 = *(_QWORD *)(v13 + 8);
      v15 = *(_DWORD *)(v13 + 24);
      if ( !v15 )
        goto LABEL_9;
      v9 = v15 - 1;
      v10 = (v15 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v11 = (__int64 *)(v14 + 16LL * v10);
      v12 = *v11;
      if ( v7 != *v11 )
      {
        v30 = 1;
        while ( v12 != -4096 )
        {
          v31 = v30 + 1;
          v10 = v9 & (v30 + v10);
          v11 = (__int64 *)(v14 + 16LL * v10);
          v12 = *v11;
          if ( *v11 == v7 )
            goto LABEL_6;
          v30 = v31;
        }
        goto LABEL_9;
      }
LABEL_6:
      if ( v11[1] )
      {
        v7 = *(_QWORD *)(v7 + 8);
        if ( v8 == v7 )
          break;
      }
      else
      {
LABEL_9:
        sub_2F59030((__int64)&v40, (__int64)a1, v7);
        LODWORD(v36) = (_DWORD)v40 + v36;
        DWORD1(v36) += HIDWORD(v40);
        *((float *)&v37 + 2) = *((float *)&v37 + 2) + *(float *)v44.m128i_i32;
        DWORD2(v36) += v41;
        HIDWORD(v36) += v42;
        *((float *)&v37 + 3) = *((float *)&v37 + 3) + *(float *)&v44.m128i_i32[1];
        LODWORD(v37) = v43 + v37;
        DWORD1(v37) += HIDWORD(v43);
        *(float *)&v38 = *(float *)&v38 + *(float *)&v44.m128i_i32[2];
        *((float *)&v38 + 1) = *((float *)&v38 + 1) + *(float *)&v44.m128i_i32[3];
        v39 = v39 + *(float *)&v45;
        v7 = *(_QWORD *)(v7 + 8);
        if ( v8 == v7 )
          break;
      }
    }
  }
  if ( DWORD1(v37) | DWORD2(v36) | (unsigned int)v37 | HIDWORD(v36) | DWORD1(v36) | (unsigned int)v36 )
  {
    v16 = (__int64 **)a1[102];
    v17 = sub_B2BE50(**v16);
    if ( sub_B6EA50(v17)
      || (v32 = sub_B2BE50(**v16),
          v33 = sub_B6F970(v32),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v33 + 48LL))(v33)) )
    {
      v18 = (__int64 *)a1[96];
      v34 = 0;
      v19 = sub_B92180(*v18);
      v20 = v19;
      if ( v19 )
      {
        v21 = *(_DWORD *)(v19 + 16);
        v22 = *(_QWORD *)(v19 + 8);
        v23 = (__int64 *)(v22 & 0xFFFFFFFFFFFFFFF8LL);
        if ( (v22 & 4) != 0 )
          v23 = (__int64 *)*v23;
        v24 = sub_B01860(v23, v21, 1u, v20, 0, 0, 0, 1);
        sub_B10CB0(&v40, (__int64)v24);
        v34 = v40;
        if ( v40 )
          sub_B976B0((__int64)&v40, v40, (__int64)&v34);
      }
      v25 = *(_QWORD *)(a1[96] + 328LL);
      sub_B157E0((__int64)&v35, &v34);
      v26 = **(_QWORD **)(v25 + 32);
      v44 = _mm_loadu_si128(&v35);
      v41 = 20;
      v43 = v26;
      v45 = "regalloc";
      v46 = "SpillReloadCopies";
      v50 = 0x400000000LL;
      LOBYTE(v42) = 2;
      v48 = 0;
      v40 = (unsigned __int8 *)&unk_4A27410;
      v47 = 17;
      v49 = (unsigned __int64 *)v51;
      v51[320] = 0;
      v52 = -1;
      v53 = v25;
      sub_2F551D0((__int64)&v36, (__int64)&v40);
      sub_B18290((__int64)&v40, "generated in function", 0x15u);
      if ( v34 )
        sub_B91220((__int64)&v34, (__int64)v34);
      sub_2EAFC50(v16, (__int64)&v40);
      v27 = v49;
      v40 = (unsigned __int8 *)&unk_49D9D40;
      v28 = &v49[10 * (unsigned int)v50];
      if ( v49 != v28 )
      {
        do
        {
          v28 -= 10;
          v29 = v28[4];
          if ( (unsigned __int64 *)v29 != v28 + 6 )
            j_j___libc_free_0(v29);
          if ( (unsigned __int64 *)*v28 != v28 + 2 )
            j_j___libc_free_0(*v28);
        }
        while ( v27 != v28 );
        v28 = v49;
      }
      if ( v28 != (unsigned __int64 *)v51 )
        _libc_free((unsigned __int64)v28);
    }
  }
}
