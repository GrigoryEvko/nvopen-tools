// Function: sub_1EBEF10
// Address: 0x1ebef10
//
__int64 __fastcall sub_1EBEF10(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v5; // r12
  __int64 v6; // r15
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 (*v14)(); // rcx
  __int64 v15; // rdx
  unsigned int v16; // edx
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // r14
  __int64 (__fastcall *v20)(__int64, __int64); // rax
  _DWORD *v21; // r15
  int v22; // eax
  int v23; // r15d
  __int64 v24; // r13
  __int64 v25; // rax
  __int64 v26; // rsi
  int v27; // eax
  _DWORD *v28; // r12
  __int64 v29; // r12
  __int64 v30; // rax
  _QWORD *v31; // r14
  __int64 v32; // rdi
  _DWORD *v33; // rax
  __int64 v34; // rdi
  signed __int64 v35; // r9
  int v36; // r8d
  int v37; // r9d
  __int64 v38; // rax
  unsigned __int64 v39; // rdx
  __int64 v40; // rdi
  unsigned __int64 v41; // r15
  unsigned __int64 v42; // rax
  int v43; // r8d
  __int64 v44; // r13
  _DWORD *v45; // r12
  _DWORD *v46; // rax
  __int64 v47; // rax
  _QWORD *v48; // rcx
  _QWORD *i; // rax
  __int64 v50; // rax
  __int64 v51; // rdx
  _QWORD *v52; // rcx
  _QWORD *j; // rax
  __int64 v54; // [rsp+10h] [rbp-160h]
  int v55; // [rsp+1Ch] [rbp-154h]
  int v57; // [rsp+20h] [rbp-150h]
  unsigned __int64 v58; // [rsp+28h] [rbp-148h]
  unsigned __int64 v59; // [rsp+28h] [rbp-148h]
  unsigned __int64 v60[2]; // [rsp+30h] [rbp-140h] BYREF
  _BYTE v61[32]; // [rsp+40h] [rbp-130h] BYREF
  _QWORD v62[2]; // [rsp+60h] [rbp-110h] BYREF
  __int64 v63; // [rsp+70h] [rbp-100h]
  __int64 v64; // [rsp+78h] [rbp-F8h]
  __int64 v65; // [rsp+80h] [rbp-F0h]
  __int64 v66; // [rsp+88h] [rbp-E8h]
  __int64 v67; // [rsp+90h] [rbp-E0h]
  __int64 v68; // [rsp+98h] [rbp-D8h]
  unsigned int v69; // [rsp+A0h] [rbp-D0h]
  char v70; // [rsp+A4h] [rbp-CCh]
  __int64 v71; // [rsp+A8h] [rbp-C8h]
  __int64 v72; // [rsp+B0h] [rbp-C0h]
  _BYTE *v73; // [rsp+B8h] [rbp-B8h]
  _BYTE *v74; // [rsp+C0h] [rbp-B0h]
  __int64 v75; // [rsp+C8h] [rbp-A8h]
  int v76; // [rsp+D0h] [rbp-A0h]
  _BYTE v77[32]; // [rsp+D8h] [rbp-98h] BYREF
  __int64 v78; // [rsp+F8h] [rbp-78h]
  _BYTE *v79; // [rsp+100h] [rbp-70h]
  _BYTE *v80; // [rsp+108h] [rbp-68h]
  __int64 v81; // [rsp+110h] [rbp-60h]
  int v82; // [rsp+118h] [rbp-58h]
  _BYTE v83[80]; // [rsp+120h] [rbp-50h] BYREF

  v5 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 248) + 24LL) + 16LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF))
     & 0xFFFFFFFFFFFFFFF8LL;
  v6 = *(_QWORD *)(a1 + 280) + 24LL * *(unsigned __int16 *)(*(_QWORD *)v5 + 24LL);
  if ( *(_DWORD *)(a1 + 288) != *(_DWORD *)v6 )
    sub_1ED7890(a1 + 280);
  if ( !*(_BYTE *)(v6 + 8) )
    return 0;
  v8 = *(_QWORD *)(a1 + 680);
  v63 = a3;
  v9 = *(_QWORD *)(a1 + 256);
  v10 = *(_QWORD *)(a1 + 264);
  v11 = a1 + 672;
  v62[0] = &unk_4A00C10;
  v65 = v10;
  v62[1] = a2;
  v12 = *(_QWORD *)(v8 + 40);
  v66 = v9;
  v64 = v12;
  v13 = *(_QWORD *)(v8 + 16);
  v14 = *(__int64 (**)())(*(_QWORD *)v13 + 40LL);
  v15 = 0;
  if ( v14 != sub_1D00B00 )
  {
    v50 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v14)(v13, v11, 0);
    v11 = a1 + 672;
    v15 = v50;
    v12 = v64;
  }
  v67 = v15;
  v16 = *(_DWORD *)(a3 + 8);
  v68 = v11;
  v69 = v16;
  v73 = v77;
  v74 = v77;
  v79 = v83;
  v80 = v83;
  v70 = 0;
  v71 = a1 + 376;
  v72 = 0;
  v75 = 4;
  v76 = 0;
  v78 = 0;
  v81 = 4;
  v82 = 0;
  *(_QWORD *)(v12 + 8) = v62;
  sub_1F15B50(*(_QWORD *)(a1 + 992), v62, 1);
  v17 = *(_QWORD *)(a1 + 984);
  v58 = *(unsigned int *)(v17 + 208);
  if ( v58 <= 1 )
    goto LABEL_37;
  v18 = *(_QWORD *)(a1 + 696);
  v19 = *(_QWORD *)(v17 + 200);
  v20 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v18 + 160LL);
  if ( v20 != sub_1E693B0 )
    v5 = ((__int64 (__fastcall *)(__int64, unsigned __int64, _QWORD))v20)(v18, v5, *(_QWORD *)(a1 + 680));
  v21 = (_DWORD *)(*(_QWORD *)(a1 + 704) + 24LL * *(unsigned __int16 *)(*(_QWORD *)v5 + 24LL));
  if ( *(_DWORD *)(a1 + 712) != *v21 )
    sub_1ED7890(a1 + 704);
  v22 = v21[1];
  v54 = v5;
  v23 = 0;
  v24 = v19;
  v55 = v22;
  v25 = 0;
  do
  {
    while ( 1 )
    {
      v31 = (_QWORD *)(v24 + 8 * v25);
      if ( (*v31 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v32 = *(_QWORD *)((*v31 & 0xFFFFFFFFFFFFFFF8LL) + 16);
        if ( v32 )
          break;
      }
LABEL_18:
      sub_1F15650(*(_QWORD *)(a1 + 992));
      v29 = sub_1F1B1B0(*(_QWORD *)(a1 + 992), *v31);
      v30 = sub_1F1BC20(*(_QWORD *)(a1 + 992), *v31);
      sub_1F1FDD0(*(_QWORD *)(a1 + 992), v29, v30);
LABEL_19:
      v25 = (unsigned int)++v23;
      if ( v58 == v23 )
        goto LABEL_26;
    }
    if ( **(_WORD **)(v32 + 16) != 15
      || (v33 = *(_DWORD **)(v32 + 32), (*v33 & 0xFFF00) != 0)
      || (v33[10] & 0xFFF00) != 0 )
    {
      v26 = sub_1E16FE0(v32, *(_DWORD *)(a2 + 112), v54, *(_QWORD *)(a1 + 688), *(_QWORD *)(a1 + 696), 1);
      v27 = 0;
      if ( v26 )
      {
        v28 = (_DWORD *)(*(_QWORD *)(a1 + 704) + 24LL * *(unsigned __int16 *)(*(_QWORD *)v26 + 24LL));
        if ( *(_DWORD *)(a1 + 712) != *v28 )
          sub_1ED7890(a1 + 704);
        v27 = v28[1];
      }
      if ( v55 == v27 )
        goto LABEL_19;
      goto LABEL_18;
    }
    v25 = (unsigned int)++v23;
  }
  while ( v58 != v23 );
LABEL_26:
  if ( *(_DWORD *)(v63 + 8) != v69 )
  {
    v34 = *(_QWORD *)(a1 + 992);
    v60[1] = 0x800000000LL;
    v60[0] = (unsigned __int64)v61;
    sub_1F1E080(v34, v60);
    sub_1DADA60(
      *(_QWORD *)(a1 + 856),
      *(_DWORD *)(a2 + 112),
      *(_QWORD *)v63 + 4LL * v69,
      *(unsigned int *)(v63 + 8) - (unsigned __int64)v69,
      *(_QWORD *)(a1 + 264),
      v35);
    v38 = *(_QWORD *)(a1 + 248);
    v39 = *(unsigned int *)(a1 + 928);
    v40 = a1 + 920;
    v41 = *(unsigned int *)(v38 + 32);
    if ( v41 < v39 )
    {
      *(_DWORD *)(a1 + 928) = v41;
      goto LABEL_29;
    }
    if ( v41 <= v39 )
    {
      v44 = *(_QWORD *)v63 + 4LL * *(unsigned int *)(v63 + 8);
      v45 = (_DWORD *)(*(_QWORD *)v63 + 4LL * v69);
    }
    else
    {
      if ( v41 > *(unsigned int *)(a1 + 932) )
      {
        sub_16CD150(v40, (const void *)(a1 + 936), v41, 8, v36, v37);
        v39 = *(unsigned int *)(a1 + 928);
        v40 = a1 + 920;
      }
      v47 = *(_QWORD *)(a1 + 920);
      v48 = (_QWORD *)(v47 + 8 * v41);
      for ( i = (_QWORD *)(v47 + 8 * v39); v48 != i; ++i )
      {
        if ( i )
          *i = *(_QWORD *)(a1 + 936);
      }
      *(_DWORD *)(a1 + 928) = v41;
      v38 = *(_QWORD *)(a1 + 248);
LABEL_29:
      v42 = *(unsigned int *)(v38 + 32);
      v43 = v42;
      v44 = *(_QWORD *)v63 + 4LL * *(unsigned int *)(v63 + 8);
      v45 = (_DWORD *)(*(_QWORD *)v63 + 4LL * v69);
      if ( v41 <= v42 )
      {
        if ( v41 < v42 )
        {
          if ( v42 > *(unsigned int *)(a1 + 932) )
          {
            v57 = v42;
            v59 = v42;
            sub_16CD150(v40, (const void *)(a1 + 936), v42, 8, v42, v37);
            v41 = *(unsigned int *)(a1 + 928);
            v43 = v57;
            v42 = v59;
          }
          v51 = *(_QWORD *)(a1 + 920);
          v52 = (_QWORD *)(v51 + 8 * v42);
          for ( j = (_QWORD *)(v51 + 8 * v41); v52 != j; ++j )
          {
            if ( j )
              *j = *(_QWORD *)(a1 + 936);
          }
          goto LABEL_30;
        }
      }
      else
      {
LABEL_30:
        *(_DWORD *)(a1 + 928) = v43;
      }
    }
    for ( ; v45 != (_DWORD *)v44; ++v45 )
    {
      v46 = (_DWORD *)(*(_QWORD *)(a1 + 920) + 8LL * (*v45 & 0x7FFFFFFF));
      if ( !*v46 )
        *v46 = 4;
    }
    if ( (_BYTE *)v60[0] != v61 )
      _libc_free(v60[0]);
  }
LABEL_37:
  *(_QWORD *)(v64 + 8) = 0;
  if ( v80 != v79 )
    _libc_free((unsigned __int64)v80);
  if ( v74 != v73 )
    _libc_free((unsigned __int64)v74);
  return 0;
}
