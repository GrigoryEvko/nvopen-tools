// Function: sub_34BA1B0
// Address: 0x34ba1b0
//
__int64 __fastcall sub_34BA1B0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  unsigned int v4; // r15d
  __int64 v5; // rax
  int v6; // eax
  __int64 v7; // r8
  unsigned __int64 v8; // r9
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 (*v11)(); // rax
  __int64 v12; // r13
  __int64 v13; // r15
  __int64 v14; // r13
  unsigned __int64 v15; // r12
  unsigned int v16; // ebx
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // r9
  __int64 v20; // r10
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  _QWORD *v23; // rax
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  _BYTE *v28; // rdi
  __int64 v29; // rax
  unsigned __int64 v30; // rdx
  __int64 *v31; // r13
  __int64 *v32; // rbx
  __int64 v33; // rdx
  __int64 *v34; // r13
  __int64 *v35; // rbx
  __int64 v36; // rdx
  __int64 *v37; // r13
  __int64 *v38; // rbx
  __int64 v39; // rdx
  _BYTE *v40; // r13
  _BYTE *v41; // rbx
  int v42; // esi
  __int64 v43; // rdx
  unsigned __int64 v44; // [rsp+8h] [rbp-338h]
  unsigned __int64 v45; // [rsp+8h] [rbp-338h]
  __int64 v46; // [rsp+18h] [rbp-328h]
  unsigned int v47; // [rsp+20h] [rbp-320h]
  __int64 v48; // [rsp+20h] [rbp-320h]
  unsigned int v49; // [rsp+4Ch] [rbp-2F4h]
  __int64 *v50; // [rsp+50h] [rbp-2F0h] BYREF
  __int64 v51; // [rsp+58h] [rbp-2E8h]
  _BYTE v52[128]; // [rsp+60h] [rbp-2E0h] BYREF
  __int64 *v53; // [rsp+E0h] [rbp-260h] BYREF
  __int64 v54; // [rsp+E8h] [rbp-258h]
  _BYTE v55[128]; // [rsp+F0h] [rbp-250h] BYREF
  __int64 *v56; // [rsp+170h] [rbp-1D0h] BYREF
  __int64 v57; // [rsp+178h] [rbp-1C8h]
  _BYTE v58[128]; // [rsp+180h] [rbp-1C0h] BYREF
  _BYTE *v59; // [rsp+200h] [rbp-140h] BYREF
  __int64 v60; // [rsp+208h] [rbp-138h]
  _BYTE v61[304]; // [rsp+210h] [rbp-130h] BYREF

  v2 = a1;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  if ( !*(_BYTE *)(a2 + 579) )
    return v2;
  v4 = *(_DWORD *)(*(_QWORD *)(a2 + 328) + 24LL);
  v5 = sub_B2E500(*(_QWORD *)a2);
  v6 = sub_B2A630(v5);
  v9 = *(_QWORD *)(a2 + 16);
  v10 = 0;
  v47 = v6 - 7;
  v11 = *(__int64 (**)())(*(_QWORD *)v9 + 128LL);
  if ( v11 != sub_2DAC790 )
    v10 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v11)(v9, a2, 0);
  v12 = *(_QWORD *)(a2 + 328);
  v50 = (__int64 *)v52;
  v53 = (__int64 *)v55;
  v51 = 0x1000000000LL;
  v54 = 0x1000000000LL;
  v56 = (__int64 *)v58;
  v57 = 0x1000000000LL;
  v59 = v61;
  v60 = 0x1000000000LL;
  if ( v12 != a2 + 320 )
  {
    v49 = v4;
    v13 = v12;
    v14 = v10;
    v46 = v2;
    v15 = v44;
    v16 = v47;
    do
    {
      if ( *(_BYTE *)(v13 + 233) )
      {
        v24 = (unsigned int)v51;
        v25 = (unsigned int)v51 + 1LL;
        if ( v25 > HIDWORD(v51) )
        {
          sub_C8D5F0((__int64)&v50, v52, v25, 8u, v7, v8);
          v24 = (unsigned int)v51;
        }
        v50[v24] = v13;
        LODWORD(v51) = v51 + 1;
      }
      else if ( v16 <= 1 && *(_BYTE *)(v13 + 216) )
      {
        v29 = (unsigned int)v57;
        v30 = (unsigned int)v57 + 1LL;
        if ( v30 > HIDWORD(v57) )
        {
          sub_C8D5F0((__int64)&v56, v58, v30, 8u, v7, v8);
          v29 = (unsigned int)v57;
        }
        v56[v29] = v13;
        LODWORD(v57) = v57 + 1;
      }
      else if ( !*(_DWORD *)(v13 + 72) )
      {
        v26 = (unsigned int)v54;
        v27 = (unsigned int)v54 + 1LL;
        if ( v27 > HIDWORD(v54) )
        {
          sub_C8D5F0((__int64)&v53, v55, v27, 8u, v7, v8);
          v26 = (unsigned int)v54;
        }
        v53[v26] = v13;
        LODWORD(v54) = v54 + 1;
      }
      v17 = sub_2E313E0(v13);
      if ( v17 != v13 + 48 && *(unsigned __int16 *)(v17 + 68) == *(_DWORD *)(v14 + 72) )
      {
        v18 = *(_QWORD *)(v17 + 32);
        v19 = v49;
        v20 = *(_QWORD *)(v18 + 24);
        if ( v16 > 1 )
          v19 = *(unsigned int *)(*(_QWORD *)(v18 + 64) + 24LL);
        v21 = (unsigned int)v60;
        v8 = v15 & 0xFFFFFFFF00000000LL | v19;
        v22 = (unsigned int)v60 + 1LL;
        v15 = v8;
        if ( v22 > HIDWORD(v60) )
        {
          v45 = v8;
          v48 = v20;
          sub_C8D5F0((__int64)&v59, v61, v22, 0x10u, v7, v8);
          v21 = (unsigned int)v60;
          v8 = v45;
          v20 = v48;
        }
        v23 = &v59[16 * v21];
        *v23 = v20;
        v23[1] = v8;
        LODWORD(v60) = v60 + 1;
      }
      v13 = *(_QWORD *)(v13 + 8);
    }
    while ( a2 + 320 != v13 );
    v2 = v46;
    if ( (_DWORD)v51 )
    {
      sub_34B9D60(v46, v49, *(_QWORD *)(a2 + 328));
      v31 = v53;
      v32 = &v53[(unsigned int)v54];
      if ( v32 != v53 )
      {
        do
        {
          v33 = *v31++;
          sub_34B9D60(v46, v49, v33);
        }
        while ( v32 != v31 );
      }
      v34 = v50;
      v35 = &v50[(unsigned int)v51];
      if ( v35 != v50 )
      {
        do
        {
          v36 = *v34++;
          sub_34B9D60(v46, *(_DWORD *)(v36 + 24), v36);
        }
        while ( v35 != v34 );
      }
      v37 = v56;
      v38 = &v56[(unsigned int)v57];
      if ( v38 != v56 )
      {
        do
        {
          v39 = *v37++;
          sub_34B9D60(v46, v49, v39);
        }
        while ( v38 != v37 );
      }
      v28 = v59;
      v40 = v59;
      v41 = &v59[16 * (unsigned int)v60];
      if ( v41 == v59 )
        goto LABEL_27;
      do
      {
        v42 = *((_DWORD *)v40 + 2);
        v43 = *(_QWORD *)v40;
        v40 += 16;
        sub_34B9D60(v46, v42, v43);
      }
      while ( v41 != v40 );
    }
    v28 = v59;
LABEL_27:
    if ( v28 != v61 )
      _libc_free((unsigned __int64)v28);
  }
  if ( v56 != (__int64 *)v58 )
    _libc_free((unsigned __int64)v56);
  if ( v53 != (__int64 *)v55 )
    _libc_free((unsigned __int64)v53);
  if ( v50 != (__int64 *)v52 )
    _libc_free((unsigned __int64)v50);
  return v2;
}
