// Function: sub_2B84950
// Address: 0x2b84950
//
void __fastcall sub_2B84950(__int64 a1, _QWORD *a2, char a3)
{
  __int64 v5; // r10
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rsi
  int v12; // ecx
  __int64 v13; // rdi
  int v14; // ecx
  unsigned int v15; // edx
  __int64 *v16; // rax
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 *v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  int v25; // r12d
  __int64 i; // r14
  __int64 v27; // rsi
  unsigned int v28; // r10d
  _BYTE *v29; // r8
  unsigned __int64 v30; // rdx
  int v31; // r11d
  _QWORD *v32; // rax
  _QWORD *v33; // rdx
  int v34; // eax
  __int64 v35; // rdi
  __int64 v36; // rax
  __int64 v37; // rdi
  __int64 v38; // rbx
  unsigned __int64 *v39; // r12
  int v40; // eax
  int v41; // r11d
  __int64 v42; // rax
  int v43; // [rsp+1Ch] [rbp-204h]
  unsigned __int64 v44; // [rsp+20h] [rbp-200h]
  _BYTE *v45; // [rsp+30h] [rbp-1F0h] BYREF
  __int64 v46; // [rsp+38h] [rbp-1E8h]
  _BYTE v47[64]; // [rsp+40h] [rbp-1E0h] BYREF
  unsigned __int64 *v48; // [rsp+80h] [rbp-1A0h] BYREF
  __int64 v49; // [rsp+88h] [rbp-198h]
  _BYTE v50[192]; // [rsp+90h] [rbp-190h] BYREF
  int v51; // [rsp+150h] [rbp-D0h]
  __int64 v52; // [rsp+158h] [rbp-C8h]
  __int64 v53; // [rsp+160h] [rbp-C0h]
  __int64 v54; // [rsp+168h] [rbp-B8h]
  _QWORD *v55; // [rsp+170h] [rbp-B0h]
  __int64 v56; // [rsp+178h] [rbp-A8h]
  __int64 v57; // [rsp+180h] [rbp-A0h]
  __int64 v58; // [rsp+188h] [rbp-98h]
  __int64 v59; // [rsp+190h] [rbp-90h] BYREF
  unsigned int v60; // [rsp+198h] [rbp-88h]
  char v61; // [rsp+1F0h] [rbp-30h] BYREF

  v5 = *(_QWORD *)(a1 + 416);
  v6 = *(unsigned int *)(a1 + 8);
  v7 = *(_QWORD *)a1;
  v48 = (unsigned __int64 *)v50;
  v49 = 0x400000000LL;
  v8 = a2[413];
  v51 = 0;
  v52 = v8;
  v9 = a2[418];
  v55 = a2;
  v53 = v9;
  v54 = a2[411];
  v10 = a2[414];
  v11 = *(_QWORD *)(v5 + 40);
  v12 = *(_DWORD *)(v10 + 24);
  v13 = *(_QWORD *)(v10 + 8);
  if ( !v12 )
  {
LABEL_39:
    v18 = 0;
    goto LABEL_4;
  }
  v14 = v12 - 1;
  v15 = v14 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
  v16 = (__int64 *)(v13 + 16LL * v15);
  v17 = *v16;
  if ( v11 != *v16 )
  {
    v40 = 1;
    while ( v17 != -4096 )
    {
      v41 = v40 + 1;
      v42 = v14 & (v15 + v40);
      v15 = v42;
      v16 = (__int64 *)(v13 + 16 * v42);
      v17 = *v16;
      if ( v11 == *v16 )
        goto LABEL_3;
      v40 = v41;
    }
    goto LABEL_39;
  }
LABEL_3:
  v18 = v16[1];
LABEL_4:
  v56 = v18;
  v19 = &v59;
  v57 = 0;
  v58 = 1;
  do
  {
    *(_DWORD *)v19 = -1;
    v19 = (__int64 *)((char *)v19 + 12);
    *((_DWORD *)v19 - 2) = -1;
  }
  while ( v19 != (__int64 *)&v61 );
  v20 = v7;
  sub_2B47310((__int64)&v48, v7, v6, v5, v6, v7);
  if ( a3 )
    sub_2B82D20((__int64 *)&v48, v20, v21, v22, v23, v24);
  v25 = *(_DWORD *)(*(_QWORD *)(a1 + 416) + 4LL) & 0x7FFFFFF;
  if ( v25 )
  {
    for ( i = 0; i != v25; ++i )
    {
      v27 = (__int64)v48;
      v28 = i;
      v29 = v47;
      v45 = v47;
      v30 = LODWORD(v48[6 * (unsigned int)i + 1]);
      v46 = 0x800000000LL;
      v31 = v30;
      if ( v30 )
      {
        v32 = v47;
        if ( v30 > 8 )
        {
          v43 = v30;
          v44 = v30;
          sub_C8D5F0((__int64)&v45, v47, v30, 8u, (__int64)v47, v24);
          v29 = v45;
          v28 = i;
          v31 = v43;
          v30 = v44;
          v32 = &v45[8 * (unsigned int)v46];
        }
        v33 = &v29[8 * v30];
        if ( v33 != v32 )
        {
          do
          {
            if ( v32 )
              *v32 = 0;
            ++v32;
          }
          while ( v33 != v32 );
          v29 = v45;
        }
        LODWORD(v46) = v31;
        v27 = (__int64)v48;
      }
      v34 = *(_DWORD *)(v27 + 8);
      if ( v34 )
      {
        v35 = (unsigned int)(v34 - 1);
        v36 = 0;
        v37 = 8 * v35;
        while ( 1 )
        {
          *(_QWORD *)&v29[v36] = *(_QWORD *)(*(_QWORD *)(v27 + 48LL * (unsigned int)i) + 2 * v36);
          if ( v37 == v36 )
            break;
          v27 = (__int64)v48;
          v29 = v45;
          v36 += 8;
        }
        v29 = v45;
      }
      sub_2B42D90(a1, v28, v29, (unsigned int)v46);
      if ( v45 != v47 )
        _libc_free((unsigned __int64)v45);
    }
  }
  if ( (v58 & 1) == 0 )
    sub_C7D6A0(v59, 12LL * v60, 4);
  v38 = (__int64)v48;
  v39 = &v48[6 * (unsigned int)v49];
  if ( v48 != v39 )
  {
    do
    {
      v39 -= 6;
      if ( (unsigned __int64 *)*v39 != v39 + 2 )
        _libc_free(*v39);
    }
    while ( (unsigned __int64 *)v38 != v39 );
    v39 = v48;
  }
  if ( v39 != (unsigned __int64 *)v50 )
    _libc_free((unsigned __int64)v39);
}
