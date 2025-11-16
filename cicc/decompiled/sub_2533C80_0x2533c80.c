// Function: sub_2533C80
// Address: 0x2533c80
//
__int64 __fastcall sub_2533C80(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 *v9; // r15
  __int64 *v10; // rbx
  unsigned __int64 j; // rax
  __int64 v12; // rdi
  unsigned int v13; // ecx
  __int64 v14; // rsi
  __int64 *v15; // rbx
  __int64 *v16; // r13
  __int64 v17; // rsi
  __int64 v18; // rdi
  __int64 **v20; // rax
  __int64 **v21; // rcx
  __int64 v22; // [rsp+8h] [rbp-4F8h]
  _QWORD v25[2]; // [rsp+40h] [rbp-4C0h] BYREF
  char v26; // [rsp+50h] [rbp-4B0h]
  __int64 v27; // [rsp+60h] [rbp-4A0h] BYREF
  __int64 v28; // [rsp+68h] [rbp-498h]
  __int64 v29; // [rsp+70h] [rbp-490h]
  __int64 v30; // [rsp+78h] [rbp-488h]
  __int64 *v31; // [rsp+80h] [rbp-480h]
  __int64 i; // [rsp+88h] [rbp-478h]
  __int64 v33[2]; // [rsp+90h] [rbp-470h] BYREF
  __int64 *v34; // [rsp+A0h] [rbp-460h]
  __int64 v35; // [rsp+A8h] [rbp-458h]
  _BYTE v36[32]; // [rsp+B0h] [rbp-450h] BYREF
  __int64 *v37; // [rsp+D0h] [rbp-430h]
  __int64 v38; // [rsp+D8h] [rbp-428h]
  _QWORD v39[2]; // [rsp+E0h] [rbp-420h] BYREF
  __int64 v40; // [rsp+F0h] [rbp-410h] BYREF
  __int64 **v41; // [rsp+F8h] [rbp-408h]
  __int64 v42; // [rsp+100h] [rbp-400h]
  int v43; // [rsp+108h] [rbp-3F8h]
  char v44; // [rsp+10Ch] [rbp-3F4h]
  _BYTE v45[16]; // [rsp+110h] [rbp-3F0h] BYREF
  __int64 v46; // [rsp+120h] [rbp-3E0h] BYREF
  _BYTE *v47; // [rsp+128h] [rbp-3D8h]
  __int64 v48; // [rsp+130h] [rbp-3D0h]
  int v49; // [rsp+138h] [rbp-3C8h]
  char v50; // [rsp+13Ch] [rbp-3C4h]
  _BYTE v51[16]; // [rsp+140h] [rbp-3C0h] BYREF
  _QWORD v52[50]; // [rsp+150h] [rbp-3B0h] BYREF
  __int64 v53; // [rsp+2E0h] [rbp-220h] BYREF
  char *v54; // [rsp+2E8h] [rbp-218h]
  __int64 v55; // [rsp+2F0h] [rbp-210h]
  int v56; // [rsp+2F8h] [rbp-208h]
  char v57; // [rsp+2FCh] [rbp-204h]
  char v58; // [rsp+300h] [rbp-200h] BYREF
  _BYTE *v59; // [rsp+380h] [rbp-180h]
  __int64 v60; // [rsp+388h] [rbp-178h]
  _BYTE v61[128]; // [rsp+390h] [rbp-170h] BYREF
  _BYTE *v62; // [rsp+410h] [rbp-F0h]
  __int64 v63; // [rsp+418h] [rbp-E8h]
  _BYTE v64[128]; // [rsp+420h] [rbp-E0h] BYREF
  __int64 v65; // [rsp+4A0h] [rbp-60h]
  __int64 v66; // [rsp+4A8h] [rbp-58h]
  __int64 v67; // [rsp+4B0h] [rbp-50h]
  __int64 v68; // [rsp+4B8h] [rbp-48h]
  __int64 v69; // [rsp+4C0h] [rbp-40h]

  v4 = a3 + 24;
  v5 = sub_BC0510(a4, &unk_4F82418, a3);
  v6 = *(_QWORD *)(v4 + 8);
  v27 = 0;
  v7 = *(_QWORD *)(v5 + 8);
  v26 = 1;
  v25[1] = 0;
  v22 = v7;
  v25[0] = v7;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = v33;
  for ( i = 0; v4 != v6; v6 = *(_QWORD *)(v6 + 8) )
  {
    v8 = v6 - 56;
    if ( !v6 )
      v8 = 0;
    v53 = v8;
    sub_2519280((__int64)&v27, &v53);
  }
  v54 = &v58;
  v59 = v61;
  v60 = 0x1000000000LL;
  v63 = 0x1000000000LL;
  v62 = v64;
  v34 = (__int64 *)v36;
  v35 = 0x400000000LL;
  v53 = 0;
  v55 = 16;
  v56 = 0;
  v57 = 1;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v68 = 0;
  v69 = 0;
  v33[0] = 0;
  v33[1] = 0;
  v37 = v39;
  v38 = 0;
  v39[0] = 0;
  v39[1] = 1;
  sub_25112E0(v52, a3, (__int64)v25, v33, 0, 1);
  if ( !(_DWORD)i || !(unsigned __int8)sub_25332C0((__int64)v52, (__int64)&v27, (__int64)&v53, v22, 1) )
  {
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
    goto LABEL_7;
  }
  v40 = 0;
  v41 = (__int64 **)v45;
  v42 = 2;
  v43 = 0;
  v44 = 1;
  v46 = 0;
  v47 = v51;
  v48 = 2;
  v49 = 0;
  v50 = 1;
  sub_AE6EC0((__int64)&v40, (__int64)&qword_4FDADA8);
  if ( HIDWORD(v48) == v49 )
  {
    if ( v44 )
    {
      v20 = v41;
      v21 = &v41[HIDWORD(v42)];
      if ( v41 != v21 )
      {
        while ( *v20 != &qword_4F82400 )
        {
          if ( v21 == ++v20 )
            goto LABEL_31;
        }
        goto LABEL_32;
      }
    }
    else if ( sub_C8CA60((__int64)&v40, (__int64)&qword_4F82400) )
    {
      goto LABEL_32;
    }
  }
LABEL_31:
  sub_AE6EC0((__int64)&v40, (__int64)&unk_4F82420);
LABEL_32:
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v45, (__int64)&v40);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v51, (__int64)&v46);
  if ( !v50 )
    _libc_free((unsigned __int64)v47);
  if ( !v44 )
    _libc_free((unsigned __int64)v41);
LABEL_7:
  sub_250E960((__int64)v52);
  v9 = v34;
  v10 = &v34[(unsigned int)v35];
  if ( v34 != v10 )
  {
    for ( j = (unsigned __int64)v34; ; j = (unsigned __int64)v34 )
    {
      v12 = *v9;
      v13 = (unsigned int)((__int64)((__int64)v9 - j) >> 3) >> 7;
      v14 = 4096LL << v13;
      if ( v13 >= 0x1E )
        v14 = 0x40000000000LL;
      ++v9;
      sub_C7D6A0(v12, v14, 16);
      if ( v10 == v9 )
        break;
    }
  }
  v15 = v37;
  v16 = &v37[2 * (unsigned int)v38];
  if ( v37 != v16 )
  {
    do
    {
      v17 = v15[1];
      v18 = *v15;
      v15 += 2;
      sub_C7D6A0(v18, v17, 16);
    }
    while ( v16 != v15 );
    v16 = v37;
  }
  if ( v16 != v39 )
    _libc_free((unsigned __int64)v16);
  if ( v34 != (__int64 *)v36 )
    _libc_free((unsigned __int64)v34);
  sub_29A2B10(&v53);
  if ( v62 != v64 )
    _libc_free((unsigned __int64)v62);
  if ( v59 != v61 )
    _libc_free((unsigned __int64)v59);
  if ( !v57 )
    _libc_free((unsigned __int64)v54);
  if ( v31 != v33 )
    _libc_free((unsigned __int64)v31);
  sub_C7D6A0(v28, 8LL * (unsigned int)v30, 8);
  return a1;
}
