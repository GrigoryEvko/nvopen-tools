// Function: sub_239B360
// Address: 0x239b360
//
_QWORD *__fastcall sub_239B360(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rcx
  __int64 v5; // r8
  _QWORD *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  _QWORD *v10; // rbx
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // rsi
  _QWORD *v14; // r14
  _QWORD *v15; // r12
  __int64 v16; // rax
  __int64 v17; // rax
  char *v18; // rbx
  char *v19; // r12
  __int64 v20; // rax
  __int64 i; // rax
  __int64 v23; // rdx
  _QWORD *v24; // r12
  __int64 v25; // rdi
  _QWORD *v26; // r13
  __int64 v27; // rax
  __int64 v28; // r15
  __int64 v29; // r14
  __int64 v30; // rax
  __int64 v31; // rcx
  _QWORD *v32; // r13
  _QWORD *v33; // r12
  __int64 v34; // rax
  __int64 v36; // [rsp+18h] [rbp-238h]
  _QWORD v37[2]; // [rsp+28h] [rbp-228h] BYREF
  __int64 v38; // [rsp+38h] [rbp-218h]
  __int64 v39; // [rsp+40h] [rbp-210h]
  void *v40; // [rsp+50h] [rbp-200h]
  __int64 v41; // [rsp+58h] [rbp-1F8h] BYREF
  __int64 v42; // [rsp+60h] [rbp-1F0h]
  __int64 v43; // [rsp+68h] [rbp-1E8h]
  __int64 v44; // [rsp+70h] [rbp-1E0h]
  _QWORD v45[2]; // [rsp+80h] [rbp-1D0h] BYREF
  char *v46; // [rsp+90h] [rbp-1C0h] BYREF
  unsigned int v47; // [rsp+98h] [rbp-1B8h]
  char v48; // [rsp+A0h] [rbp-1B0h] BYREF
  __int64 v49; // [rsp+120h] [rbp-130h]
  __int64 v50; // [rsp+128h] [rbp-128h]
  __int64 v51; // [rsp+130h] [rbp-120h]
  unsigned int v52; // [rsp+138h] [rbp-118h]
  char v53; // [rsp+140h] [rbp-110h]
  void *v54; // [rsp+150h] [rbp-100h]
  __int64 v55; // [rsp+158h] [rbp-F8h] BYREF
  _QWORD *v56; // [rsp+160h] [rbp-F0h] BYREF
  __int64 v57; // [rsp+168h] [rbp-E8h]
  _QWORD v58[16]; // [rsp+170h] [rbp-E0h] BYREF
  __int64 v59; // [rsp+1F0h] [rbp-60h]
  __int64 v60; // [rsp+1F8h] [rbp-58h]
  __int64 v61; // [rsp+200h] [rbp-50h]
  unsigned int v62; // [rsp+208h] [rbp-48h]
  char v63; // [rsp+210h] [rbp-40h]

  sub_CFB860((__int64)v45, a2 + 8, a3, a4);
  v54 = (void *)v45[0];
  v55 = v45[1];
  v56 = v58;
  v57 = 0x400000000LL;
  if ( v47 )
    sub_239AF40((__int64)&v56, (__int64)&v46, v47, v4, v5);
  ++v49;
  v59 = 1;
  v60 = v50;
  v50 = 0;
  v61 = v51;
  v51 = 0;
  v62 = v52;
  v52 = 0;
  v63 = v53;
  v6 = (_QWORD *)sub_22077B0(0xD0u);
  v10 = v6;
  if ( v6 )
  {
    *v6 = &unk_4A0B1A0;
    v6[1] = v54;
    v6[2] = v55;
    v6[3] = v6 + 5;
    v6[4] = 0x400000000LL;
    if ( (_DWORD)v57 )
      sub_239AF40((__int64)(v6 + 3), (__int64)&v56, v7, v8, v9);
    v11 = v60;
    ++v59;
    v12 = 0;
    v13 = 0;
    v10[21] = 1;
    v10[22] = v11;
    v60 = 0;
    v10[23] = v61;
    v61 = 0;
    *((_DWORD *)v10 + 48) = v62;
    v62 = 0;
    *((_BYTE *)v10 + 200) = v63;
  }
  else
  {
    v12 = v60;
    v13 = 88LL * v62;
    if ( v62 )
    {
      v37[0] = 2;
      v37[1] = 0;
      v38 = -4096;
      v39 = 0;
      v41 = 2;
      v42 = 0;
      v43 = -8192;
      v44 = 0;
      v40 = &unk_49DDAE8;
      v36 = v60 + 88LL * v62;
      for ( i = -4096; ; i = v38 )
      {
        v23 = *(_QWORD *)(v12 + 24);
        if ( v23 != i )
        {
          i = v43;
          if ( v23 != v43 )
          {
            v24 = *(_QWORD **)(v12 + 40);
            v25 = 4LL * *(unsigned int *)(v12 + 48);
            v26 = &v24[v25];
            if ( v24 != &v24[v25] )
            {
              do
              {
                v27 = *(v26 - 2);
                v26 -= 4;
                if ( v27 != -4096 && v27 != 0 && v27 != -8192 )
                  sub_BD60C0(v26);
              }
              while ( v24 != v26 );
              v26 = *(_QWORD **)(v12 + 40);
            }
            if ( v26 != (_QWORD *)(v12 + 56) )
              _libc_free((unsigned __int64)v26);
            i = *(_QWORD *)(v12 + 24);
          }
        }
        *(_QWORD *)v12 = &unk_49DB368;
        if ( i != -4096 && i != 0 && i != -8192 )
          sub_BD60C0((_QWORD *)(v12 + 8));
        v12 += 88;
        if ( v36 == v12 )
          break;
      }
      v40 = &unk_49DB368;
      sub_D68D70(&v41);
      sub_D68D70(v37);
      v12 = v60;
      v13 = 88LL * v62;
    }
  }
  sub_C7D6A0(v12, v13, 8);
  v14 = v56;
  v15 = &v56[4 * (unsigned int)v57];
  if ( v56 != v15 )
  {
    do
    {
      v16 = *(v15 - 2);
      v15 -= 4;
      if ( v16 != 0 && v16 != -4096 && v16 != -8192 )
        sub_BD60C0(v15);
    }
    while ( v14 != v15 );
    v15 = v56;
  }
  if ( v15 != v58 )
    _libc_free((unsigned __int64)v15);
  *a1 = v10;
  v17 = v52;
  if ( v52 )
  {
    v41 = 2;
    v28 = v50;
    v42 = 0;
    v43 = -4096;
    v40 = &unk_49DDAE8;
    v54 = &unk_49DDAE8;
    v44 = 0;
    v55 = 2;
    v29 = v50 + 88LL * v52;
    v30 = -4096;
    v56 = 0;
    v57 = -8192;
    v58[0] = 0;
    while ( 1 )
    {
      v31 = *(_QWORD *)(v28 + 24);
      if ( v31 != v30 )
      {
        v30 = v57;
        if ( v31 != v57 )
        {
          v32 = *(_QWORD **)(v28 + 40);
          v33 = &v32[4 * *(unsigned int *)(v28 + 48)];
          if ( v32 != v33 )
          {
            do
            {
              v34 = *(v33 - 2);
              v33 -= 4;
              if ( v34 != -4096 && v34 != 0 && v34 != -8192 )
                sub_BD60C0(v33);
            }
            while ( v32 != v33 );
            v33 = *(_QWORD **)(v28 + 40);
          }
          if ( v33 != (_QWORD *)(v28 + 56) )
            _libc_free((unsigned __int64)v33);
          v30 = *(_QWORD *)(v28 + 24);
        }
      }
      *(_QWORD *)v28 = &unk_49DB368;
      if ( v30 != -4096 && v30 != 0 && v30 != -8192 )
        sub_BD60C0((_QWORD *)(v28 + 8));
      v28 += 88;
      if ( v29 == v28 )
        break;
      v30 = v43;
    }
    v54 = &unk_49DB368;
    sub_D68D70(&v55);
    v40 = &unk_49DB368;
    sub_D68D70(&v41);
    v17 = v52;
  }
  sub_C7D6A0(v50, 88 * v17, 8);
  v18 = v46;
  v19 = &v46[32 * v47];
  if ( v46 != v19 )
  {
    do
    {
      v20 = *((_QWORD *)v19 - 2);
      v19 -= 32;
      if ( v20 != -4096 && v20 != 0 && v20 != -8192 )
        sub_BD60C0(v19);
    }
    while ( v18 != v19 );
    v19 = v46;
  }
  if ( v19 != &v48 )
    _libc_free((unsigned __int64)v19);
  return a1;
}
