// Function: sub_2D46690
// Address: 0x2d46690
//
__int64 __fastcall sub_2D46690(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        __int64 a6,
        __int64 (__fastcall *a7)(__int64, __int64, __int64),
        __int64 a8)
{
  _QWORD *v9; // rbx
  __int64 v10; // rdx
  __int64 v11; // r15
  _QWORD *v12; // rax
  __int64 *v13; // rsi
  __int64 v14; // rax
  __int64 v15; // r13
  _QWORD *v16; // rdi
  _QWORD *v17; // rax
  __int64 v18; // rbx
  __int64 v19; // r15
  __int64 v20; // r14
  __int64 v21; // rdx
  unsigned int v22; // esi
  _QWORD *v23; // rbx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rbx
  __int64 v27; // rax
  _QWORD *v28; // r15
  _QWORD *v29; // rax
  __int64 v30; // rbx
  __int64 v31; // r13
  __int64 v32; // r15
  __int64 v33; // rdx
  unsigned int v34; // esi
  _QWORD **v36; // rdx
  int v37; // ecx
  __int64 *v38; // rax
  __int64 v39; // rax
  __int64 v40; // r14
  __int64 v41; // rbx
  __int64 v42; // rdx
  unsigned int v43; // esi
  __int64 v48; // [rsp+28h] [rbp-C8h]
  _QWORD *v49; // [rsp+30h] [rbp-C0h]
  __int64 v50; // [rsp+30h] [rbp-C0h]
  __int64 v51; // [rsp+48h] [rbp-A8h]
  __int64 v52; // [rsp+58h] [rbp-98h]
  const char *v53; // [rsp+60h] [rbp-90h] BYREF
  char v54; // [rsp+80h] [rbp-70h]
  char v55; // [rsp+81h] [rbp-6Fh]
  _QWORD v56[4]; // [rsp+90h] [rbp-60h] BYREF
  __int16 v57; // [rsp+B0h] [rbp-40h]

  v9 = *(_QWORD **)(a2 + 48);
  v10 = *(unsigned __int16 *)(a2 + 64);
  v11 = v9[9];
  v12 = *(_QWORD **)(a2 + 72);
  v13 = *(__int64 **)(a2 + 56);
  v56[0] = "atomicrmw.end";
  v49 = v12;
  v57 = 259;
  v51 = sub_AA8550(v9, v13, v10, (__int64)v56, 0);
  v56[0] = "atomicrmw.start";
  v57 = 259;
  v14 = sub_22077B0(0x50u);
  v15 = v14;
  if ( v14 )
    sub_AA4D50(v14, (__int64)v49, (__int64)v56, v11, v51);
  v16 = (_QWORD *)((v9[6] & 0xFFFFFFFFFFFFFFF8LL) - 24);
  if ( (v9[6] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    v16 = 0;
  sub_B43D60(v16);
  *(_QWORD *)(a2 + 48) = v9;
  *(_QWORD *)(a2 + 56) = v9 + 6;
  *(_WORD *)(a2 + 64) = 0;
  v57 = 257;
  v17 = sub_BD2C40(72, 1u);
  v18 = (__int64)v17;
  if ( v17 )
    sub_B4C8F0((__int64)v17, v15, 1u, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
    *(_QWORD *)(a2 + 88),
    v18,
    v56,
    *(_QWORD *)(a2 + 56),
    *(_QWORD *)(a2 + 64));
  v19 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v19 )
  {
    v20 = *(_QWORD *)a2;
    do
    {
      v21 = *(_QWORD *)(v20 + 8);
      v22 = *(_DWORD *)v20;
      v20 += 16;
      sub_B99FD0(v18, v22, v21);
    }
    while ( v19 != v20 );
  }
  v23 = a1;
  *(_QWORD *)(a2 + 48) = v15;
  *(_QWORD *)(a2 + 56) = v15 + 48;
  *(_WORD *)(a2 + 64) = 0;
  v48 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD))(*(_QWORD *)*a1 + 1032LL))(
          *a1,
          a2,
          a3,
          a4,
          a5);
  v24 = a7(a8, a2, v48);
  v25 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD))(*(_QWORD *)*v23 + 1040LL))(
          *v23,
          a2,
          v24,
          a4,
          a5);
  v55 = 1;
  v26 = v25;
  v54 = 3;
  v53 = "tryagain";
  v27 = sub_BCCE00(v49, 0x20u);
  v50 = sub_ACD640(v27, 0, 0);
  v28 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a2 + 80) + 56LL))(
                    *(_QWORD *)(a2 + 80),
                    33,
                    v26,
                    v50);
  if ( !v28 )
  {
    v57 = 257;
    v28 = sub_BD2C40(72, unk_3F10FD0);
    if ( v28 )
    {
      v36 = *(_QWORD ***)(v26 + 8);
      v37 = *((unsigned __int8 *)v36 + 8);
      if ( (unsigned int)(v37 - 17) > 1 )
      {
        v39 = sub_BCB2A0(*v36);
      }
      else
      {
        BYTE4(v52) = (_BYTE)v37 == 18;
        LODWORD(v52) = *((_DWORD *)v36 + 8);
        v38 = (__int64 *)sub_BCB2A0(*v36);
        v39 = sub_BCE1B0(v38, v52);
      }
      sub_B523C0((__int64)v28, v39, 53, 33, v26, v50, (__int64)v56, 0, 0, 0);
    }
    (*(void (__fastcall **)(_QWORD, _QWORD *, const char **, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
      *(_QWORD *)(a2 + 88),
      v28,
      &v53,
      *(_QWORD *)(a2 + 56),
      *(_QWORD *)(a2 + 64));
    v40 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
    if ( *(_QWORD *)a2 != v40 )
    {
      v41 = *(_QWORD *)a2;
      do
      {
        v42 = *(_QWORD *)(v41 + 8);
        v43 = *(_DWORD *)v41;
        v41 += 16;
        sub_B99FD0((__int64)v28, v43, v42);
      }
      while ( v40 != v41 );
    }
  }
  v57 = 257;
  v29 = sub_BD2C40(72, 3u);
  v30 = (__int64)v29;
  if ( v29 )
    sub_B4C9A0((__int64)v29, v15, v51, (__int64)v28, 3u, 0, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
    *(_QWORD *)(a2 + 88),
    v30,
    v56,
    *(_QWORD *)(a2 + 56),
    *(_QWORD *)(a2 + 64));
  v31 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v31 )
  {
    v32 = *(_QWORD *)a2;
    do
    {
      v33 = *(_QWORD *)(v32 + 8);
      v34 = *(_DWORD *)v32;
      v32 += 16;
      sub_B99FD0(v30, v34, v33);
    }
    while ( v31 != v32 );
  }
  sub_A88F30(a2, v51, *(_QWORD *)(v51 + 56), 1);
  return v48;
}
