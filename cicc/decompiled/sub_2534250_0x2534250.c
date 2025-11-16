// Function: sub_2534250
// Address: 0x2534250
//
__int64 __fastcall sub_2534250(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 *v11; // rax
  __int64 *v12; // rbx
  __int64 *v13; // r13
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // rax
  __int64 *v17; // r14
  __int64 *v18; // rbx
  unsigned __int64 i; // rax
  __int64 v20; // rdi
  unsigned int v21; // ecx
  __int64 v22; // rsi
  __int64 *v23; // rbx
  __int64 *v24; // r13
  __int64 v25; // rsi
  __int64 v26; // rdi
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 **v32; // rax
  __int64 **v33; // rdx
  __int64 v34; // [rsp+10h] [rbp-4F0h]
  _QWORD v39[2]; // [rsp+40h] [rbp-4C0h] BYREF
  char v40; // [rsp+50h] [rbp-4B0h]
  __int64 v41; // [rsp+60h] [rbp-4A0h] BYREF
  __int64 v42; // [rsp+68h] [rbp-498h]
  __int64 v43; // [rsp+70h] [rbp-490h]
  __int64 v44; // [rsp+78h] [rbp-488h]
  __int64 *v45; // [rsp+80h] [rbp-480h]
  __int64 v46; // [rsp+88h] [rbp-478h]
  __int64 v47[2]; // [rsp+90h] [rbp-470h] BYREF
  __int64 *v48; // [rsp+A0h] [rbp-460h]
  __int64 v49; // [rsp+A8h] [rbp-458h]
  _BYTE v50[32]; // [rsp+B0h] [rbp-450h] BYREF
  __int64 *v51; // [rsp+D0h] [rbp-430h]
  __int64 v52; // [rsp+D8h] [rbp-428h]
  _QWORD v53[2]; // [rsp+E0h] [rbp-420h] BYREF
  __int64 v54; // [rsp+F0h] [rbp-410h] BYREF
  __int64 **v55; // [rsp+F8h] [rbp-408h]
  __int64 v56; // [rsp+100h] [rbp-400h]
  int v57; // [rsp+108h] [rbp-3F8h]
  char v58; // [rsp+10Ch] [rbp-3F4h]
  _BYTE v59[16]; // [rsp+110h] [rbp-3F0h] BYREF
  __int64 v60; // [rsp+120h] [rbp-3E0h] BYREF
  _BYTE *v61; // [rsp+128h] [rbp-3D8h]
  __int64 v62; // [rsp+130h] [rbp-3D0h]
  int v63; // [rsp+138h] [rbp-3C8h]
  char v64; // [rsp+13Ch] [rbp-3C4h]
  _BYTE v65[16]; // [rsp+140h] [rbp-3C0h] BYREF
  _QWORD v66[50]; // [rsp+150h] [rbp-3B0h] BYREF
  __int64 v67; // [rsp+2E0h] [rbp-220h] BYREF
  char *v68; // [rsp+2E8h] [rbp-218h]
  __int64 v69; // [rsp+2F0h] [rbp-210h]
  int v70; // [rsp+2F8h] [rbp-208h]
  char v71; // [rsp+2FCh] [rbp-204h]
  char v72; // [rsp+300h] [rbp-200h] BYREF
  _BYTE *v73; // [rsp+380h] [rbp-180h]
  __int64 v74; // [rsp+388h] [rbp-178h]
  _BYTE v75[128]; // [rsp+390h] [rbp-170h] BYREF
  _BYTE *v76; // [rsp+410h] [rbp-F0h]
  __int64 v77; // [rsp+418h] [rbp-E8h]
  _BYTE v78[128]; // [rsp+420h] [rbp-E0h] BYREF
  __int64 v79; // [rsp+4A0h] [rbp-60h]
  __int64 v80; // [rsp+4A8h] [rbp-58h]
  __int64 v81; // [rsp+4B0h] [rbp-50h]
  __int64 v82; // [rsp+4B8h] [rbp-48h]
  __int64 v83; // [rsp+4C0h] [rbp-40h]

  v8 = sub_227ED20(a4, &qword_4FDADA8, (__int64 *)a3, a5);
  v9 = *(unsigned int *)(a3 + 16);
  v41 = 0;
  v10 = *(_QWORD *)(v8 + 8);
  v40 = 0;
  v39[1] = 0;
  v34 = v10;
  v39[0] = v10;
  v45 = v47;
  v11 = *(__int64 **)(a3 + 8);
  v42 = 0;
  v12 = &v11[v9];
  v13 = v11;
  v43 = 0;
  v44 = 0;
  v46 = 0;
  if ( v11 == v12 )
    goto LABEL_25;
  do
  {
    v14 = *v13++;
    v67 = *(_QWORD *)(v14 + 8);
    sub_2519280((__int64)&v41, &v67);
  }
  while ( v12 != v13 );
  if ( !(_DWORD)v46 )
  {
LABEL_25:
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 16) = 2;
    *(_DWORD *)(a1 + 24) = 0;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    sub_AE6EC0(a1, (__int64)&qword_4F82400);
    goto LABEL_26;
  }
  v15 = *(_QWORD *)(v45[(unsigned int)v46 - 1] + 40);
  v76 = v78;
  v68 = &v72;
  v73 = v75;
  v74 = 0x1000000000LL;
  v77 = 0x1000000000LL;
  v79 = a5;
  v80 = a3;
  v81 = a4;
  v67 = 0;
  v69 = 16;
  v70 = 0;
  v71 = 1;
  v83 = 0;
  v82 = a6;
  v16 = *(_QWORD *)(sub_227ED20(a4, &qword_4FDADA8, (__int64 *)a3, a5) + 8);
  v47[0] = 0;
  v83 = v16;
  v48 = (__int64 *)v50;
  v49 = 0x400000000LL;
  v51 = v53;
  v47[1] = 0;
  v52 = 0;
  v53[0] = 0;
  v53[1] = 1;
  sub_25112E0(v66, v15, (__int64)v39, v47, (__int64)&v41, 1);
  if ( (_DWORD)v46 && (unsigned __int8)sub_25332C0((__int64)v66, (__int64)&v41, (__int64)&v67, v34, 0) )
  {
    v54 = 0;
    v61 = v65;
    v55 = (__int64 **)v59;
    v56 = 2;
    v57 = 0;
    v58 = 1;
    v60 = 0;
    v62 = 2;
    v63 = 0;
    v64 = 1;
    sub_2508EA0((__int64)&v54, (__int64)&qword_4FDADA8, v28, v29, v30, v31);
    if ( HIDWORD(v62) == v63 )
    {
      if ( v58 )
      {
        v32 = v55;
        v33 = &v55[HIDWORD(v56)];
        if ( v55 != v33 )
        {
          while ( *v32 != &qword_4F82400 )
          {
            if ( v33 == ++v32 )
              goto LABEL_31;
          }
          goto LABEL_32;
        }
      }
      else if ( sub_C8CA60((__int64)&v54, (__int64)&qword_4F82400) )
      {
        goto LABEL_32;
      }
    }
LABEL_31:
    sub_AE6EC0((__int64)&v54, (__int64)&unk_4F82420);
LABEL_32:
    sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v59, (__int64)&v54);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v65, (__int64)&v60);
    if ( !v64 )
      _libc_free((unsigned __int64)v61);
    if ( !v58 )
      _libc_free((unsigned __int64)v55);
    goto LABEL_6;
  }
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 16) = 2;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  sub_AE6EC0(a1, (__int64)&qword_4F82400);
LABEL_6:
  sub_250E960((__int64)v66);
  v17 = v48;
  v18 = &v48[(unsigned int)v49];
  if ( v48 != v18 )
  {
    for ( i = (unsigned __int64)v48; ; i = (unsigned __int64)v48 )
    {
      v20 = *v17;
      v21 = (unsigned int)((__int64)((__int64)v17 - i) >> 3) >> 7;
      v22 = 4096LL << v21;
      if ( v21 >= 0x1E )
        v22 = 0x40000000000LL;
      ++v17;
      sub_C7D6A0(v20, v22, 16);
      if ( v18 == v17 )
        break;
    }
  }
  v23 = v51;
  v24 = &v51[2 * (unsigned int)v52];
  if ( v51 != v24 )
  {
    do
    {
      v25 = v23[1];
      v26 = *v23;
      v23 += 2;
      sub_C7D6A0(v26, v25, 16);
    }
    while ( v24 != v23 );
    v24 = v51;
  }
  if ( v24 != v53 )
    _libc_free((unsigned __int64)v24);
  if ( v48 != (__int64 *)v50 )
    _libc_free((unsigned __int64)v48);
  sub_29A2B10(&v67);
  if ( v76 != v78 )
    _libc_free((unsigned __int64)v76);
  if ( v73 != v75 )
    _libc_free((unsigned __int64)v73);
  if ( !v71 )
    _libc_free((unsigned __int64)v68);
LABEL_26:
  if ( v45 != v47 )
    _libc_free((unsigned __int64)v45);
  sub_C7D6A0(v42, 8LL * (unsigned int)v44, 8);
  return a1;
}
