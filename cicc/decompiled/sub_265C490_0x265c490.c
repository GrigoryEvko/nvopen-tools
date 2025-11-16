// Function: sub_265C490
// Address: 0x265c490
//
__int64 __fastcall sub_265C490(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5)
{
  unsigned int v6; // r12d
  _BYTE **v7; // rbx
  int v8; // r13d
  const char *v9; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 result; // rax
  __int64 v14; // rdx
  __int32 v15; // esi
  __int64 v16; // r12
  __int64 v17; // rbx
  __int64 v18; // rdx
  __int64 v19; // r9
  __int64 v20; // rbx
  bool v21; // zf
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // r12
  unsigned __int8 *v25; // rax
  __int64 v26; // r12
  __int64 v27; // rax
  unsigned __int64 *v28; // rbx
  unsigned __int64 *v29; // r12
  unsigned __int64 v30; // rdi
  _QWORD *v31; // rbx
  char v32; // al
  __int64 v33; // r9
  unsigned __int64 v34; // rbx
  unsigned int v35; // esi
  int v36; // eax
  int v37; // eax
  unsigned __int64 v38; // rax
  unsigned __int64 v39; // [rsp+8h] [rbp-328h]
  __int64 v40; // [rsp+8h] [rbp-328h]
  __int64 v42; // [rsp+18h] [rbp-318h]
  const char *v43; // [rsp+20h] [rbp-310h]
  unsigned __int8 *v45; // [rsp+38h] [rbp-2F8h]
  __int64 v46; // [rsp+38h] [rbp-2F8h]
  __int64 v47; // [rsp+38h] [rbp-2F8h]
  __int64 v48; // [rsp+38h] [rbp-2F8h]
  __int64 v49; // [rsp+38h] [rbp-2F8h]
  _QWORD *v50; // [rsp+38h] [rbp-2F8h]
  __int64 v51; // [rsp+40h] [rbp-2F0h]
  __int64 *v52; // [rsp+40h] [rbp-2F0h]
  __int64 v53; // [rsp+50h] [rbp-2E0h]
  unsigned __int64 v55[2]; // [rsp+60h] [rbp-2D0h] BYREF
  __int64 v56; // [rsp+70h] [rbp-2C0h] BYREF
  __int64 *v57; // [rsp+80h] [rbp-2B0h]
  __int64 v58; // [rsp+90h] [rbp-2A0h] BYREF
  unsigned __int64 v59[2]; // [rsp+B0h] [rbp-280h] BYREF
  __int64 v60; // [rsp+C0h] [rbp-270h] BYREF
  __int64 *v61; // [rsp+D0h] [rbp-260h]
  __int64 v62; // [rsp+E0h] [rbp-250h] BYREF
  _QWORD *v63; // [rsp+100h] [rbp-230h] BYREF
  unsigned __int64 v64; // [rsp+108h] [rbp-228h] BYREF
  _QWORD v65[2]; // [rsp+110h] [rbp-220h] BYREF
  __int64 *v66; // [rsp+120h] [rbp-210h]
  __int64 v67; // [rsp+130h] [rbp-200h] BYREF
  const char *v68; // [rsp+150h] [rbp-1E0h] BYREF
  __int64 v69; // [rsp+158h] [rbp-1D8h] BYREF
  __int64 v70; // [rsp+160h] [rbp-1D0h] BYREF
  __int64 v71; // [rsp+168h] [rbp-1C8h]
  __int64 v72; // [rsp+170h] [rbp-1C0h]
  unsigned __int64 *v73; // [rsp+1A0h] [rbp-190h]
  unsigned int v74; // [rsp+1A8h] [rbp-188h]
  char v75; // [rsp+1B0h] [rbp-180h] BYREF

  v6 = *(_DWORD *)(a2 + 16);
  if ( v6 != 1 )
  {
    v7 = *(_BYTE ***)a1;
    if ( !***(_BYTE ***)a1 )
    {
      sub_264B5B0((__int64)&v68, (unsigned __int64)v7[2], v6, (__int64)v7[3], (__int64 *)v7[4], v7[5], a5);
      sub_265C280((__int64)v7[1], (__int64)&v68);
      sub_2649CB0((__int64)v68, (__int64)&v68[8 * (unsigned int)v69]);
      if ( v68 != (const char *)&v70 )
        _libc_free((unsigned __int64)v68);
      *v7[6] = 1;
      **v7 = 1;
      *(_DWORD *)v7[7] = v6;
    }
  }
  v8 = 0;
  v9 = sub_BD5D20(a4);
  v42 = v12;
  v43 = v9;
  result = 0;
  if ( *(_DWORD *)(a2 + 16) )
  {
    while ( 1 )
    {
      v14 = *(_QWORD *)(a2 + 8);
      v15 = *(_DWORD *)(v14 + 4 * result);
      if ( v15 )
        break;
LABEL_37:
      result = (unsigned int)(v8 + 1);
      v8 = result;
      if ( *(_DWORD *)(a2 + 16) <= (unsigned int)result )
        return result;
    }
    v16 = *(_QWORD *)(a1 + 8);
    v69 = v42;
    v17 = *(_QWORD *)(a4 + 24);
    LOWORD(v72) = 261;
    v68 = v43;
    sub_2644DA0((__int64 *)&v63, v15, v14, v42, v10, v11, a5);
    v53 = sub_BA8CA0(v16, (__int64)v63, v64, v17);
    v19 = v18;
    if ( v63 != v65 )
    {
      v51 = v18;
      j_j___libc_free_0((unsigned __int64)v63);
      v19 = v51;
    }
    if ( !v8 )
    {
      v20 = a3;
      goto LABEL_9;
    }
    v31 = *(_QWORD **)(**(_QWORD **)(a1 + 16) + 8LL * (unsigned int)(v8 - 1));
    v64 = 2;
    v65[1] = a3;
    v65[0] = 0;
    if ( a3 != 0 && a3 != -4096 && a3 != -8192 )
    {
      v46 = v19;
      sub_BD73F0((__int64)&v64);
      v19 = v46;
    }
    v47 = v19;
    v66 = v31;
    v63 = &unk_49DD7B0;
    v32 = sub_F9E960((__int64)v31, (__int64)&v63, v55);
    v33 = v47;
    if ( v32 )
    {
      v34 = v55[0] + 40;
LABEL_44:
      v48 = v33;
      v63 = &unk_49DB368;
      sub_D68D70(&v64);
      v20 = *(_QWORD *)(v34 + 16);
      v19 = v48;
LABEL_9:
      v21 = *(_QWORD *)(v20 - 32) == 0;
      *(_QWORD *)(v20 + 80) = v53;
      if ( !v21 )
      {
        v22 = *(_QWORD *)(v20 - 24);
        **(_QWORD **)(v20 - 16) = v22;
        if ( v22 )
          *(_QWORD *)(v22 + 16) = *(_QWORD *)(v20 - 16);
      }
      *(_QWORD *)(v20 - 32) = v19;
      if ( v19 )
      {
        v23 = *(_QWORD *)(v19 + 16);
        *(_QWORD *)(v20 - 24) = v23;
        if ( v23 )
          *(_QWORD *)(v23 + 16) = v20 - 24;
        *(_QWORD *)(v20 - 16) = v19 + 16;
        *(_QWORD *)(v19 + 16) = v20 - 32;
      }
      v45 = (unsigned __int8 *)v19;
      v52 = *(__int64 **)(a1 + 24);
      sub_B174A0((__int64)&v68, (__int64)"memprof-context-disambiguation", (__int64)"MemprofCall", 11, v20);
      sub_B16080((__int64)v55, "Call", 4, (unsigned __int8 *)v20);
      v24 = sub_2647050((__int64)&v68, (__int64)v55);
      sub_B18290(v24, " in clone ", 0xAu);
      v25 = (unsigned __int8 *)sub_B43CB0(v20);
      sub_B16080((__int64)v59, "Caller", 6, v25);
      v26 = sub_23FD640(v24, (__int64)v59);
      sub_B18290(v26, " assigned to call function clone ", 0x21u);
      sub_B16080((__int64)&v63, "Callee", 6, v45);
      v27 = sub_23FD640(v26, (__int64)&v63);
      sub_1049740(v52, v27);
      if ( v66 != &v67 )
        j_j___libc_free_0((unsigned __int64)v66);
      if ( v63 != v65 )
        j_j___libc_free_0((unsigned __int64)v63);
      if ( v61 != &v62 )
        j_j___libc_free_0((unsigned __int64)v61);
      if ( (__int64 *)v59[0] != &v60 )
        j_j___libc_free_0(v59[0]);
      if ( v57 != &v58 )
        j_j___libc_free_0((unsigned __int64)v57);
      if ( (__int64 *)v55[0] != &v56 )
        j_j___libc_free_0(v55[0]);
      v28 = v73;
      v68 = (const char *)&unk_49D9D40;
      v29 = &v73[10 * v74];
      if ( v73 != v29 )
      {
        do
        {
          v29 -= 10;
          v30 = v29[4];
          if ( (unsigned __int64 *)v30 != v29 + 6 )
            j_j___libc_free_0(v30);
          if ( (unsigned __int64 *)*v29 != v29 + 2 )
            j_j___libc_free_0(*v29);
        }
        while ( v28 != v29 );
        v29 = v73;
      }
      if ( v29 != (unsigned __int64 *)&v75 )
        _libc_free((unsigned __int64)v29);
      goto LABEL_37;
    }
    v59[0] = v55[0];
    v35 = *((_DWORD *)v31 + 6);
    v36 = *((_DWORD *)v31 + 4);
    ++*v31;
    v37 = v36 + 1;
    if ( 4 * v37 >= 3 * v35 )
    {
      sub_CF32C0((__int64)v31, 2 * v35);
    }
    else
    {
      if ( v35 - *((_DWORD *)v31 + 5) - v37 > v35 >> 3 )
        goto LABEL_47;
      sub_CF32C0((__int64)v31, v35);
    }
    sub_F9E960((__int64)v31, (__int64)&v63, v59);
    v33 = v47;
    v37 = *((_DWORD *)v31 + 4) + 1;
LABEL_47:
    *((_DWORD *)v31 + 4) = v37;
    v38 = v59[0];
    v69 = 2;
    v70 = 0;
    v71 = -4096;
    v72 = 0;
    if ( *(_QWORD *)(v59[0] + 24) != -4096 )
    {
      --*((_DWORD *)v31 + 5);
      v68 = (const char *)&unk_49DB368;
      if ( v71 != -4096 && v71 != 0 && v71 != -8192 )
      {
        v39 = v38;
        v49 = v33;
        sub_BD60C0(&v69);
        v38 = v39;
        v33 = v49;
      }
    }
    v40 = v33;
    v50 = (_QWORD *)v38;
    sub_263F860((unsigned __int64 *)(v38 + 8), &v64);
    v33 = v40;
    v50[4] = v66;
    v34 = (unsigned __int64)(v50 + 5);
    v50[5] = 6;
    v50[6] = 0;
    v50[7] = 0;
    goto LABEL_44;
  }
  return result;
}
