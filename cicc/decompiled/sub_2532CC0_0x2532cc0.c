// Function: sub_2532CC0
// Address: 0x2532cc0
//
__int64 __fastcall sub_2532CC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 *v11; // rax
  __int64 *v12; // rbx
  __int64 *v13; // r14
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // rax
  __int64 *v17; // r15
  __int64 *v18; // rbx
  unsigned __int64 i; // rdx
  __int64 v20; // rdi
  unsigned int v21; // ecx
  __int64 v22; // rsi
  __int64 *v23; // rbx
  __int64 *v24; // r14
  __int64 v25; // rsi
  __int64 v26; // rdi
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r9
  void *v31; // [rsp+8h] [rbp-508h]
  void *v32; // [rsp+10h] [rbp-500h]
  _QWORD v37[2]; // [rsp+50h] [rbp-4C0h] BYREF
  char v38; // [rsp+60h] [rbp-4B0h]
  __int64 v39; // [rsp+70h] [rbp-4A0h] BYREF
  __int64 v40; // [rsp+78h] [rbp-498h]
  __int64 v41; // [rsp+80h] [rbp-490h]
  __int64 v42; // [rsp+88h] [rbp-488h]
  __int64 *v43; // [rsp+90h] [rbp-480h]
  __int64 v44; // [rsp+98h] [rbp-478h]
  __int64 v45[2]; // [rsp+A0h] [rbp-470h] BYREF
  __int64 *v46; // [rsp+B0h] [rbp-460h]
  __int64 v47; // [rsp+B8h] [rbp-458h]
  _BYTE v48[32]; // [rsp+C0h] [rbp-450h] BYREF
  __int64 *v49; // [rsp+E0h] [rbp-430h]
  __int64 v50; // [rsp+E8h] [rbp-428h]
  _QWORD v51[2]; // [rsp+F0h] [rbp-420h] BYREF
  __int64 v52; // [rsp+100h] [rbp-410h] BYREF
  _BYTE *v53; // [rsp+108h] [rbp-408h]
  __int64 v54; // [rsp+110h] [rbp-400h]
  int v55; // [rsp+118h] [rbp-3F8h]
  char v56; // [rsp+11Ch] [rbp-3F4h]
  _BYTE v57[16]; // [rsp+120h] [rbp-3F0h] BYREF
  __int64 v58; // [rsp+130h] [rbp-3E0h] BYREF
  _BYTE *v59; // [rsp+138h] [rbp-3D8h]
  __int64 v60; // [rsp+140h] [rbp-3D0h]
  int v61; // [rsp+148h] [rbp-3C8h]
  char v62; // [rsp+14Ch] [rbp-3C4h]
  _BYTE v63[16]; // [rsp+150h] [rbp-3C0h] BYREF
  _QWORD v64[50]; // [rsp+160h] [rbp-3B0h] BYREF
  __int64 v65; // [rsp+2F0h] [rbp-220h] BYREF
  char *v66; // [rsp+2F8h] [rbp-218h]
  __int64 v67; // [rsp+300h] [rbp-210h]
  int v68; // [rsp+308h] [rbp-208h]
  char v69; // [rsp+30Ch] [rbp-204h]
  char v70; // [rsp+310h] [rbp-200h] BYREF
  _BYTE *v71; // [rsp+390h] [rbp-180h]
  __int64 v72; // [rsp+398h] [rbp-178h]
  _BYTE v73[128]; // [rsp+3A0h] [rbp-170h] BYREF
  _BYTE *v74; // [rsp+420h] [rbp-F0h]
  __int64 v75; // [rsp+428h] [rbp-E8h]
  _BYTE v76[128]; // [rsp+430h] [rbp-E0h] BYREF
  __int64 v77; // [rsp+4B0h] [rbp-60h]
  __int64 v78; // [rsp+4B8h] [rbp-58h]
  __int64 v79; // [rsp+4C0h] [rbp-50h]
  __int64 v80; // [rsp+4C8h] [rbp-48h]
  __int64 v81; // [rsp+4D0h] [rbp-40h]

  v8 = sub_227ED20(a4, &qword_4FDADA8, (__int64 *)a3, a5);
  v9 = *(unsigned int *)(a3 + 16);
  v39 = 0;
  v10 = *(_QWORD *)(v8 + 8);
  v38 = 0;
  v37[1] = 0;
  v37[0] = v10;
  v43 = v45;
  v11 = *(__int64 **)(a3 + 8);
  v40 = 0;
  v12 = &v11[v9];
  v13 = v11;
  v41 = 0;
  v42 = 0;
  v44 = 0;
  if ( v11 == v12 )
    goto LABEL_25;
  do
  {
    v14 = *v13++;
    v65 = *(_QWORD *)(v14 + 8);
    sub_2519280((__int64)&v39, &v65);
  }
  while ( v12 != v13 );
  if ( !(_DWORD)v44 )
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
  }
  else
  {
    v31 = (void *)(a1 + 32);
    v32 = (void *)(a1 + 80);
    v15 = *(_QWORD *)(v43[(unsigned int)v44 - 1] + 40);
    v74 = v76;
    v66 = &v70;
    v71 = v73;
    v72 = 0x1000000000LL;
    v75 = 0x1000000000LL;
    v77 = a5;
    v78 = a3;
    v79 = a4;
    v65 = 0;
    v67 = 16;
    v68 = 0;
    v69 = 1;
    v81 = 0;
    v80 = a6;
    v16 = *(_QWORD *)(sub_227ED20(a4, &qword_4FDADA8, (__int64 *)a3, a5) + 8);
    v45[0] = 0;
    v81 = v16;
    v46 = (__int64 *)v48;
    v47 = 0x400000000LL;
    v49 = v51;
    v45[1] = 0;
    v50 = 0;
    v51[0] = 0;
    v51[1] = 1;
    sub_25112E0(v64, v15, (__int64)v37, v45, (__int64)&v39, 1);
    if ( (_DWORD)v44 && (unsigned __int8)sub_2532010((__int64)v64, (__int64)&v39, (__int64)&v65, 0, 0) )
    {
      v52 = 0;
      v53 = v57;
      v54 = 2;
      v55 = 0;
      v56 = 1;
      v58 = 0;
      v59 = v63;
      v60 = 2;
      v61 = 0;
      v62 = 1;
      sub_2508EA0((__int64)&v52, (__int64)&qword_4FDADA8, v28, v29, (__int64)&v52, v30);
      sub_C8CF70(a1, v31, 2, (__int64)v57, (__int64)&v52);
      sub_C8CF70(a1 + 48, v32, 2, (__int64)v63, (__int64)&v58);
      if ( !v62 )
        _libc_free((unsigned __int64)v59);
      if ( !v56 )
        _libc_free((unsigned __int64)v53);
    }
    else
    {
      *(_BYTE *)(a1 + 28) = 1;
      *(_QWORD *)a1 = 0;
      *(_QWORD *)(a1 + 8) = v31;
      *(_QWORD *)(a1 + 16) = 2;
      *(_DWORD *)(a1 + 24) = 0;
      *(_QWORD *)(a1 + 48) = 0;
      *(_QWORD *)(a1 + 56) = v32;
      *(_QWORD *)(a1 + 64) = 2;
      *(_DWORD *)(a1 + 72) = 0;
      *(_BYTE *)(a1 + 76) = 1;
      sub_AE6EC0(a1, (__int64)&qword_4F82400);
    }
    sub_250E960((__int64)v64);
    v17 = v46;
    v18 = &v46[(unsigned int)v47];
    if ( v46 != v18 )
    {
      for ( i = (unsigned __int64)v46; ; i = (unsigned __int64)v46 )
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
    v23 = v49;
    v24 = &v49[2 * (unsigned int)v50];
    if ( v49 != v24 )
    {
      do
      {
        v25 = v23[1];
        v26 = *v23;
        v23 += 2;
        sub_C7D6A0(v26, v25, 16);
      }
      while ( v24 != v23 );
      v24 = v49;
    }
    if ( v24 != v51 )
      _libc_free((unsigned __int64)v24);
    if ( v46 != (__int64 *)v48 )
      _libc_free((unsigned __int64)v46);
    sub_29A2B10(&v65);
    if ( v74 != v76 )
      _libc_free((unsigned __int64)v74);
    if ( v71 != v73 )
      _libc_free((unsigned __int64)v71);
    if ( !v69 )
      _libc_free((unsigned __int64)v66);
  }
  if ( v43 != v45 )
    _libc_free((unsigned __int64)v43);
  sub_C7D6A0(v40, 8LL * (unsigned int)v42, 8);
  return a1;
}
