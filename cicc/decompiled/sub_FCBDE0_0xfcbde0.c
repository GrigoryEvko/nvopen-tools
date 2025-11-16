// Function: sub_FCBDE0
// Address: 0xfcbde0
//
void __fastcall sub_FCBDE0(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r14
  unsigned __int8 *v9; // rbx
  unsigned __int8 *v10; // r14
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r14
  __int64 v14; // rbx
  __int64 v15; // rax
  bool v16; // zf
  unsigned int **v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rcx
  unsigned int *v20; // rbx
  unsigned int *v21; // r14
  __int64 v22; // r15
  __int64 *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rdi
  int v26; // edx
  __int64 v27; // rcx
  __int64 v28; // r14
  __int64 v29; // rax
  unsigned __int64 v30; // rdx
  __int64 v31; // rdx
  _QWORD *v32; // rbx
  _QWORD *v33; // r15
  __int64 v34; // rax
  __int64 v35; // r8
  __int64 v36; // rdx
  unsigned __int64 v37; // r9
  unsigned int v38; // edx
  unsigned int v39; // eax
  __int64 v40; // r14
  unsigned int **v41; // r15
  bool v42; // bl
  __int64 *v43; // rax
  unsigned __int8 v44; // cl
  unsigned int v45; // ebx
  unsigned __int64 v46; // rax
  __int64 v47; // rdx
  int v48; // r15d
  __int64 v49; // rax
  unsigned int **v50; // rdi
  unsigned int *v51; // rdi
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rsi
  __int64 v55; // rax
  int v56; // eax
  __int64 v57; // rax
  int v58; // edx
  __int64 *v59; // [rsp+8h] [rbp-108h]
  __int64 v60; // [rsp+20h] [rbp-F0h]
  __int64 v61; // [rsp+20h] [rbp-F0h]
  unsigned __int64 v62; // [rsp+38h] [rbp-D8h] BYREF
  __int64 v63; // [rsp+40h] [rbp-D0h] BYREF
  unsigned __int64 v64; // [rsp+48h] [rbp-C8h] BYREF
  __int64 *v65; // [rsp+50h] [rbp-C0h]
  __int64 v66; // [rsp+58h] [rbp-B8h]
  unsigned int **v67; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v68; // [rsp+68h] [rbp-A8h]
  _BYTE v69[32]; // [rsp+70h] [rbp-A0h] BYREF
  unsigned int *v70; // [rsp+90h] [rbp-80h] BYREF
  __int64 v71; // [rsp+98h] [rbp-78h]
  _BYTE v72[112]; // [rsp+A0h] [rbp-70h] BYREF

  v8 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
  if ( (a2[7] & 0x40) != 0 )
  {
    v9 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
    v10 = &v9[v8];
  }
  else
  {
    v9 = &a2[-v8];
    v10 = a2;
  }
  for ( ; v10 != v9; v9 += 32 )
  {
    v11 = sub_FC8800(a1, *(_QWORD *)v9, a3, a4, a5, a6);
    if ( v11 )
    {
      if ( *(_QWORD *)v9 )
      {
        v12 = *((_QWORD *)v9 + 1);
        **((_QWORD **)v9 + 2) = v12;
        if ( v12 )
          *(_QWORD *)(v12 + 16) = *((_QWORD *)v9 + 2);
      }
      *(_QWORD *)v9 = v11;
      a3 = *(_QWORD *)(v11 + 16);
      a4 = v11 + 16;
      *((_QWORD *)v9 + 1) = a3;
      if ( a3 )
        *(_QWORD *)(a3 + 16) = v9 + 8;
      *((_QWORD *)v9 + 2) = a4;
      *(_QWORD *)(v11 + 16) = v9;
    }
  }
  if ( *a2 == 84 && (*((_DWORD *)a2 + 1) & 0x7FFFFFF) != 0 )
  {
    v13 = 0;
    v14 = 8LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
    do
    {
      v15 = sub_FC8800(a1, *(_QWORD *)(*((_QWORD *)a2 - 1) + 32LL * *((unsigned int *)a2 + 18) + v13), a3, a4, a5, a6);
      if ( v15 )
      {
        a3 = *((_QWORD *)a2 - 1) + 32LL * *((unsigned int *)a2 + 18);
        *(_QWORD *)(a3 + v13) = v15;
      }
      v13 += 8;
    }
    while ( v14 != v13 );
  }
  v16 = *((_QWORD *)a2 + 6) == 0;
  v70 = (unsigned int *)v72;
  v71 = 0x400000000LL;
  if ( !v16 || (a2[7] & 0x20) != 0 )
  {
    v17 = &v70;
    sub_B9AA80((__int64)a2, (__int64)&v70);
    v20 = v70;
    v21 = &v70[4 * (unsigned int)v71];
    if ( v21 != v70 )
    {
      do
      {
        v22 = *((_QWORD *)v20 + 1);
        v17 = (unsigned int **)v22;
        v23 = sub_FC95E0(a1, v22, v18, v19, a5, a6);
        v66 = v24;
        v65 = v23;
        if ( (_BYTE)v24 )
        {
          v18 = (__int64)v65;
        }
        else
        {
          v17 = (unsigned int **)v22;
          v18 = sub_FCBA10(a1, v22, v24, v19, a5, a6);
        }
        if ( v22 != v18 )
        {
          v17 = (unsigned int **)*v20;
          sub_B99FD0((__int64)a2, (unsigned int)v17, v18);
        }
        v20 += 4;
      }
      while ( v21 != v20 );
    }
    v25 = *(_QWORD *)(a1 + 8);
    if ( !v25 )
      goto LABEL_45;
LABEL_28:
    v26 = *a2;
    if ( (unsigned __int8)(v26 - 34) <= 0x33u )
    {
      v27 = 0x8000000000041LL;
      if ( _bittest64(&v27, (unsigned int)(v26 - 34)) )
      {
        v28 = *((_QWORD *)a2 + 10);
        v67 = (unsigned int **)v69;
        v68 = 0x300000000LL;
        v29 = *(unsigned int *)(v28 + 12);
        v30 = (unsigned int)(v29 - 1);
        if ( v30 > 3 )
        {
          sub_C8D5F0((__int64)&v67, v69, v30, 8u, a5, a6);
          v29 = *(unsigned int *)(v28 + 12);
          v25 = *(_QWORD *)(a1 + 8);
        }
        v31 = *(_QWORD *)(v28 + 16);
        v32 = (_QWORD *)(v31 + 8 * v29);
        if ( v32 == (_QWORD *)(v31 + 8) )
        {
          v38 = v68;
        }
        else
        {
          v33 = (_QWORD *)(v31 + 8);
          do
          {
            v34 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v25 + 24LL))(v25, *v33);
            v36 = (unsigned int)v68;
            v37 = (unsigned int)v68 + 1LL;
            if ( v37 > HIDWORD(v68) )
            {
              v61 = v34;
              sub_C8D5F0((__int64)&v67, v69, (unsigned int)v68 + 1LL, 8u, v35, v37);
              v36 = (unsigned int)v68;
              v34 = v61;
            }
            ++v33;
            v67[v36] = (unsigned int *)v34;
            v25 = *(_QWORD *)(a1 + 8);
            v38 = v68 + 1;
            LODWORD(v68) = v68 + 1;
          }
          while ( v32 != v33 );
        }
        v39 = *(_DWORD *)(v28 + 8);
        v40 = v38;
        v41 = v67;
        v42 = v39 >> 8 != 0;
        v43 = (__int64 *)(*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v25 + 24LL))(v25, *((_QWORD *)a2 + 1));
        v44 = v42;
        v17 = v41;
        v45 = 0;
        v46 = sub_BCF480(v43, v41, v40, v44);
        v47 = **(_QWORD **)(v46 + 16);
        *((_QWORD *)a2 + 10) = v46;
        *((_QWORD *)a2 + 1) = v47;
        v59 = (__int64 *)sub_BD5C60((__int64)a2);
        v62 = *((_QWORD *)a2 + 9);
        if ( (unsigned int)sub_A74480((__int64)&v62) )
        {
          do
          {
            v48 = 80;
            while ( 1 )
            {
              v17 = (unsigned int **)v45;
              v64 = sub_A747F0(&v62, v45, v48);
              v49 = sub_A72A60((__int64 *)&v64);
              if ( v49 )
                break;
              if ( ++v48 == 86 )
              {
                ++v45;
                goto LABEL_42;
              }
            }
            v60 = (*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 8) + 24LL))(*(_QWORD *)(a1 + 8), v49);
            v63 = sub_A747F0(&v62, v45, v48);
            v64 = sub_A7B980((__int64 *)&v62, v59, v45, v48);
            v56 = sub_A71AE0(&v63);
            v57 = sub_A77D30(v59, v56, v60);
            v58 = v45;
            v17 = (unsigned int **)v59;
            ++v45;
            v62 = sub_A7B440((__int64 *)&v64, v59, v58, v57);
LABEL_42:
            ;
          }
          while ( (unsigned int)sub_A74480((__int64)&v62) > v45 );
        }
        v50 = v67;
        *((_QWORD *)a2 + 9) = v62;
        if ( v50 != (unsigned int **)v69 )
          _libc_free(v50, v17);
LABEL_45:
        v51 = v70;
        if ( v70 == (unsigned int *)v72 )
          return;
        goto LABEL_46;
      }
      if ( (_BYTE)v26 == 60 )
      {
        v52 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v25 + 24LL))(v25, *((_QWORD *)a2 + 9));
        LOBYTE(v26) = *a2;
        *((_QWORD *)a2 + 9) = v52;
        v25 = *(_QWORD *)(a1 + 8);
      }
      if ( (_BYTE)v26 == 63 )
      {
        v53 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v25 + 24LL))(v25, *((_QWORD *)a2 + 9));
        v54 = *((_QWORD *)a2 + 10);
        *((_QWORD *)a2 + 9) = v53;
        *((_QWORD *)a2 + 10) = (*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 8) + 24LL))(
                                 *(_QWORD *)(a1 + 8),
                                 v54);
        v25 = *(_QWORD *)(a1 + 8);
      }
    }
    v17 = (unsigned int **)*((_QWORD *)a2 + 1);
    v55 = (*(__int64 (__fastcall **)(__int64, unsigned int **))(*(_QWORD *)v25 + 24LL))(v25, v17);
    v51 = v70;
    *((_QWORD *)a2 + 1) = v55;
    if ( v51 == (unsigned int *)v72 )
      return;
LABEL_46:
    _libc_free(v51, v17);
    return;
  }
  v25 = *(_QWORD *)(a1 + 8);
  if ( v25 )
    goto LABEL_28;
}
