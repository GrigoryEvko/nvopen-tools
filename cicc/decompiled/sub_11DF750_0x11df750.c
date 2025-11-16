// Function: sub_11DF750
// Address: 0x11df750
//
__int64 __fastcall sub_11DF750(__int64 a1, __int64 a2, __int64 a3)
{
  int v6; // edx
  __int64 v7; // r15
  unsigned __int64 v8; // rax
  _QWORD *v9; // r10
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 result; // rax
  __int64 v13; // rdi
  _QWORD *v14; // r10
  _BYTE *v15; // rax
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // r11
  __int64 v18; // rax
  __int64 v19; // r15
  __int64 v20; // rax
  char v21; // al
  _QWORD *v22; // rax
  __int64 v23; // r9
  __int64 v24; // r13
  __int64 v25; // r14
  unsigned int *v26; // r12
  unsigned int *v27; // r14
  __int64 v28; // rdx
  unsigned int v29; // esi
  _QWORD **v30; // r13
  unsigned int v31; // eax
  __int64 v32; // rax
  __int64 v33; // rax
  unsigned __int8 *v34; // rax
  size_t v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  char v38; // al
  unsigned int *v39; // r14
  __int64 v40; // rdx
  unsigned int v41; // esi
  _BYTE *v42; // rax
  _QWORD *v43; // rdi
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // r15
  __int64 v47; // rax
  char v48; // al
  _QWORD *v49; // rax
  __int64 v50; // r13
  __int64 v51; // r14
  unsigned int *v52; // r12
  unsigned int *v53; // r14
  __int64 v54; // rdx
  unsigned int v55; // esi
  size_t v56; // rax
  __int64 v57; // [rsp-10h] [rbp-E0h]
  unsigned __int64 v58; // [rsp+0h] [rbp-D0h]
  unsigned __int64 v59; // [rsp+8h] [rbp-C8h]
  __int64 v60; // [rsp+8h] [rbp-C8h]
  _BYTE *v61; // [rsp+10h] [rbp-C0h]
  __int64 *v62; // [rsp+10h] [rbp-C0h]
  char v63; // [rsp+10h] [rbp-C0h]
  __int64 v64; // [rsp+10h] [rbp-C0h]
  _QWORD *v65; // [rsp+18h] [rbp-B8h]
  __int64 v66; // [rsp+18h] [rbp-B8h]
  char v67; // [rsp+18h] [rbp-B8h]
  char v68; // [rsp+18h] [rbp-B8h]
  size_t v69; // [rsp+20h] [rbp-B0h]
  __int64 v70; // [rsp+20h] [rbp-B0h]
  _QWORD *v71; // [rsp+20h] [rbp-B0h]
  __int64 v72; // [rsp+28h] [rbp-A8h]
  __int64 v73; // [rsp+28h] [rbp-A8h]
  _BYTE *v74; // [rsp+38h] [rbp-98h] BYREF
  void *s; // [rsp+40h] [rbp-90h] BYREF
  size_t n; // [rsp+48h] [rbp-88h]
  __m128i v77; // [rsp+50h] [rbp-80h] BYREF
  __int64 v78; // [rsp+60h] [rbp-70h]
  __int64 v79; // [rsp+68h] [rbp-68h]
  __int64 v80; // [rsp+70h] [rbp-60h]
  __int64 v81; // [rsp+78h] [rbp-58h]
  __int64 v82; // [rsp+80h] [rbp-50h]
  __int64 v83; // [rsp+88h] [rbp-48h]
  __int16 v84; // [rsp+90h] [rbp-40h]

  v6 = *(_DWORD *)(a2 + 4);
  v78 = 0;
  v79 = 0;
  v80 = 0;
  v81 = 0;
  v7 = *(_QWORD *)(a2 + 32 * (2LL - (v6 & 0x7FFFFFF)));
  v8 = *(_QWORD *)(a1 + 16);
  v82 = 0;
  v84 = 257;
  v77 = (__m128i)v8;
  v83 = 0;
  if ( (unsigned __int8)sub_9B6260(v7, &v77, 0) )
  {
    v77.m128i_i32[0] = 0;
    sub_11DA4B0(a2, v77.m128i_i32, 1);
  }
  v77.m128i_i32[0] = 1;
  sub_11DA4B0(a2, v77.m128i_i32, 1);
  if ( *(_BYTE *)v7 != 17 )
    return 0;
  v9 = *(_QWORD **)(v7 + 24);
  if ( *(_DWORD *)(v7 + 32) > 0x40u )
    v9 = (_QWORD *)*v9;
  v72 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v10 = 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v11 = *(_QWORD *)(a2 + v10);
  if ( (unsigned __int64)v9 <= 1 )
  {
    if ( v9 == (_QWORD *)1 )
    {
      v36 = sub_BCB2B0(*(_QWORD **)(a3 + 72));
      v70 = sub_ACD640(v36, 0, 0);
      v37 = sub_AA4E30(*(_QWORD *)(a3 + 48));
      v60 = v70;
      v38 = sub_AE5020(v37, *(_QWORD *)(v70 + 8));
      LOWORD(v80) = 257;
      v63 = v38;
      v71 = sub_BD2C40(80, unk_3F10A10);
      if ( v71 )
        sub_B4D3C0((__int64)v71, v60, v72, 0, v63, v60, 0, 0);
      (*(void (__fastcall **)(_QWORD, _QWORD *, __m128i *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
        *(_QWORD *)(a3 + 88),
        v71,
        &v77,
        *(_QWORD *)(a3 + 56),
        *(_QWORD *)(a3 + 64));
      v39 = *(unsigned int **)a3;
      v73 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
      if ( *(_QWORD *)a3 != v73 )
      {
        do
        {
          v40 = *((_QWORD *)v39 + 1);
          v41 = *v39;
          v39 += 4;
          sub_B99FD0((__int64)v71, v41, v40);
        }
        while ( (unsigned int *)v73 != v39 );
      }
    }
    result = sub_11CA050(v11, a3, *(_QWORD *)(a1 + 16), *(__int64 **)(a1 + 24));
    if ( result )
    {
      if ( *(_BYTE *)result == 85 )
        *(_WORD *)(result + 2) = *(_WORD *)(result + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
      return result;
    }
    return 0;
  }
  v13 = *(_QWORD *)(a2 + v10);
  v65 = v9;
  s = 0;
  n = 0;
  if ( !(unsigned __int8)sub_98B0F0(v13, &s, 0) )
    return 0;
  v14 = v65;
  v69 = n;
  if ( !n || (v61 = s, v15 = memchr(s, 0, n), v14 = v65, !v15) )
  {
    v17 = -1;
LABEL_27:
    v35 = (size_t)v14 - 1;
    if ( (unsigned __int64)v14 - 1 > v69 )
      v35 = v69;
    v66 = v35;
    goto LABEL_18;
  }
  v16 = v15 - v61;
  v17 = v16;
  if ( (unsigned __int64)v65 <= v16 )
  {
    v56 = v69;
    if ( v69 > v17 )
      v56 = v17;
    v69 = v56;
    goto LABEL_27;
  }
  v69 = v16;
  v66 = v16 + 1;
LABEL_18:
  if ( v69 )
  {
    v58 = v17;
    v59 = (unsigned __int64)v14;
    v62 = *(__int64 **)(a1 + 24);
    v30 = (_QWORD **)sub_B43CA0(a2);
    v31 = sub_97FA80(*v62, (__int64)v30);
    v32 = sub_BCCE00(*v30, v31);
    v33 = sub_ACD640(v32, v66, 0);
    v34 = (unsigned __int8 *)sub_B343C0(a3, 0xEEu, v72, 0x100u, v11, 0x100u, v33, 0, 0, 0, 0, 0);
    sub_11DAF00(v34, a2);
    if ( v59 <= v58 )
    {
      v42 = (_BYTE *)sub_AD64C0(*(_QWORD *)(a2 + 8), v66, 0);
      v43 = *(_QWORD **)(a3 + 72);
      LOWORD(v80) = 257;
      v74 = v42;
      v44 = sub_BCB2B0(v43);
      v64 = sub_921130((unsigned int **)a3, v44, v72, &v74, 1, (__int64)&v77, 3u);
      v45 = sub_BCB2B0(*(_QWORD **)(a3 + 72));
      v46 = sub_ACD640(v45, 0, 0);
      v47 = sub_AA4E30(*(_QWORD *)(a3 + 48));
      v48 = sub_AE5020(v47, *(_QWORD *)(v46 + 8));
      LOWORD(v80) = 257;
      v68 = v48;
      v49 = sub_BD2C40(80, unk_3F10A10);
      v50 = (__int64)v49;
      if ( v49 )
        sub_B4D3C0((__int64)v49, v46, v64, 0, v68, v64, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, __m128i *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
        *(_QWORD *)(a3 + 88),
        v50,
        &v77,
        *(_QWORD *)(a3 + 56),
        *(_QWORD *)(a3 + 64));
      v51 = 4LL * *(unsigned int *)(a3 + 8);
      v52 = *(unsigned int **)a3;
      v53 = &v52[v51];
      while ( v53 != v52 )
      {
        v54 = *((_QWORD *)v52 + 1);
        v55 = *v52;
        v52 += 4;
        sub_B99FD0(v50, v55, v54);
      }
    }
    return sub_AD64C0(*(_QWORD *)(a2 + 8), v69, 0);
  }
  else
  {
    v18 = sub_BCB2B0(*(_QWORD **)(a3 + 72));
    v19 = sub_ACD640(v18, 0, 0);
    v20 = sub_AA4E30(*(_QWORD *)(a3 + 48));
    v21 = sub_AE5020(v20, *(_QWORD *)(v19 + 8));
    LOWORD(v80) = 257;
    v67 = v21;
    v22 = sub_BD2C40(80, unk_3F10A10);
    v24 = (__int64)v22;
    if ( v22 )
    {
      sub_B4D3C0((__int64)v22, v19, v72, 0, v67, v23, 0, 0);
      v23 = v57;
    }
    (*(void (__fastcall **)(_QWORD, __int64, __m128i *, _QWORD, _QWORD, __int64))(**(_QWORD **)(a3 + 88) + 16LL))(
      *(_QWORD *)(a3 + 88),
      v24,
      &v77,
      *(_QWORD *)(a3 + 56),
      *(_QWORD *)(a3 + 64),
      v23);
    v25 = 4LL * *(unsigned int *)(a3 + 8);
    v26 = *(unsigned int **)a3;
    v27 = &v26[v25];
    while ( v27 != v26 )
    {
      v28 = *((_QWORD *)v26 + 1);
      v29 = *v26;
      v26 += 4;
      sub_B99FD0(v24, v29, v28);
    }
    return sub_AD64C0(*(_QWORD *)(a2 + 8), 0, 0);
  }
}
