// Function: sub_1DCD8B0
// Address: 0x1dcd8b0
//
__int64 __fastcall sub_1DCD8B0(_QWORD *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  _QWORD *v6; // r14
  unsigned __int16 v8; // bx
  __int64 v9; // r12
  __int16 **v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 result; // rax
  __int16 v14; // dx
  __int16 *v15; // r12
  __int64 v16; // rax
  __int64 v17; // rdx
  __int16 *v18; // rax
  __int16 v19; // dx
  __int16 *v20; // rax
  __int16 *v21; // rbx
  _DWORD *v22; // rax
  _DWORD *v23; // rcx
  _DWORD *v24; // rax
  _BYTE *v25; // rcx
  __int16 v26; // ax
  unsigned int v27; // eax
  __int64 v28; // rax
  int *v29; // rdi
  __int64 v30; // rsi
  __int64 v31; // rcx
  __int64 v32; // rax
  int *v33; // rdi
  __int64 v34; // rsi
  __int64 v35; // rcx
  int v36; // r8d
  __int64 v37; // rdx
  unsigned __int16 *v38; // rax
  __int64 v39; // rcx
  unsigned __int16 *v40; // r14
  int v41; // r12d
  int v42; // eax
  __int64 v43; // [rsp+8h] [rbp-158h]
  __int64 v45; // [rsp+30h] [rbp-130h]
  unsigned __int16 v46; // [rsp+48h] [rbp-118h]
  unsigned __int16 v47; // [rsp+4Eh] [rbp-112h]
  __int64 v48; // [rsp+50h] [rbp-110h] BYREF
  int v49; // [rsp+58h] [rbp-108h]
  __int64 v50; // [rsp+60h] [rbp-100h]
  __int64 v51; // [rsp+68h] [rbp-F8h]
  __int64 v52; // [rsp+70h] [rbp-F0h]
  _BYTE *v53; // [rsp+80h] [rbp-E0h] BYREF
  __int64 v54; // [rsp+88h] [rbp-D8h]
  _BYTE v55[24]; // [rsp+90h] [rbp-D0h] BYREF
  int v56; // [rsp+A8h] [rbp-B8h] BYREF
  __int64 v57; // [rsp+B0h] [rbp-B0h]
  int *v58; // [rsp+B8h] [rbp-A8h]
  int *v59; // [rsp+C0h] [rbp-A0h]
  __int64 v60; // [rsp+C8h] [rbp-98h]
  unsigned __int64 v61; // [rsp+D0h] [rbp-90h] BYREF
  __int64 v62; // [rsp+D8h] [rbp-88h]
  __int64 v63; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v64; // [rsp+E8h] [rbp-78h]
  __int64 v65; // [rsp+F0h] [rbp-70h]
  int v66; // [rsp+108h] [rbp-58h] BYREF
  __int64 v67; // [rsp+110h] [rbp-50h]
  int *v68; // [rsp+118h] [rbp-48h]
  int *v69; // [rsp+120h] [rbp-40h]
  __int64 v70; // [rsp+128h] [rbp-38h]

  v5 = a2;
  v6 = a1;
  v8 = a2;
  v9 = *(_QWORD *)(a1[46] + 8LL * a2);
  v10 = (__int16 **)(a1[49] + 8LL * a2);
  if ( v9 )
  {
    if ( !*v10 )
    {
      v27 = sub_1E16810(*(_QWORD *)(a1[46] + 8LL * a2), a2, 0, 0, 0);
      if ( v27 == -1 || !(*(_QWORD *)(v9 + 32) + 40LL * v27) )
      {
        v61 = 805306368;
        v63 = 0;
        LODWORD(v62) = a2;
        v64 = 0;
        v65 = 0;
        sub_1E1AFD0(v9, &v61);
      }
    }
    goto LABEL_3;
  }
  v15 = *v10;
  if ( !*v10 )
  {
    v56 = 0;
    v53 = v55;
    v54 = 0x400000000LL;
    v57 = 0;
    v58 = &v56;
    v59 = &v56;
    v60 = 0;
    v16 = sub_1DCD430((__int64)a1, a2, (__int64)&v53, a4, a5);
    if ( !v16 )
      goto LABEL_32;
    v45 = v16;
    v61 = 805306368;
    v63 = 0;
    LODWORD(v62) = a2;
    v64 = 0;
    v65 = 0;
    sub_1E1AFD0(v16, &v61);
    *(_QWORD *)(a1[46] + 8LL * a2) = v45;
    v17 = a1[45];
    v61 = (unsigned __int64)&v63;
    v62 = 0x800000000LL;
    v66 = 0;
    v67 = 0;
    v68 = &v66;
    v69 = &v66;
    v70 = 0;
    if ( !v17 )
      BUG();
    v43 = a2;
    v46 = a2;
    v18 = (__int16 *)(*(_QWORD *)(v17 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v17 + 8) + 24LL * a2 + 4));
    v19 = *v18;
    v20 = v18 + 1;
    if ( v19 )
      v15 = v20;
    v47 = v19 + a2;
    v21 = v15;
    while ( 1 )
    {
      if ( !v21 )
      {
        v5 = v43;
        v6 = a1;
        v8 = v46;
        sub_1DCADB0(v67);
        if ( (__int64 *)v61 != &v63 )
          _libc_free(v61);
LABEL_32:
        sub_1DCADB0(v57);
        if ( v53 != v55 )
          _libc_free((unsigned __int64)v53);
        break;
      }
      if ( v70 )
      {
        v28 = v67;
        if ( v67 )
        {
          v29 = &v66;
          do
          {
            while ( 1 )
            {
              v30 = *(_QWORD *)(v28 + 16);
              v31 = *(_QWORD *)(v28 + 24);
              if ( (unsigned int)v47 <= *(_DWORD *)(v28 + 32) )
                break;
              v28 = *(_QWORD *)(v28 + 24);
              if ( !v31 )
                goto LABEL_42;
            }
            v29 = (int *)v28;
            v28 = *(_QWORD *)(v28 + 16);
          }
          while ( v30 );
LABEL_42:
          if ( v29 != &v66 && v47 >= (unsigned int)v29[8] )
            goto LABEL_28;
        }
      }
      else
      {
        v22 = (_DWORD *)v61;
        v23 = (_DWORD *)(v61 + 4LL * (unsigned int)v62);
        if ( (_DWORD *)v61 != v23 )
        {
          while ( v47 != *v22 )
          {
            if ( v23 == ++v22 )
              goto LABEL_22;
          }
          if ( v23 != v22 )
            goto LABEL_28;
        }
      }
LABEL_22:
      if ( v60 )
      {
        v32 = v57;
        if ( !v57 )
          goto LABEL_53;
        v33 = &v56;
        do
        {
          while ( 1 )
          {
            v34 = *(_QWORD *)(v32 + 16);
            v35 = *(_QWORD *)(v32 + 24);
            if ( (unsigned int)v47 <= *(_DWORD *)(v32 + 32) )
              break;
            v32 = *(_QWORD *)(v32 + 24);
            if ( !v35 )
              goto LABEL_51;
          }
          v33 = (int *)v32;
          v32 = *(_QWORD *)(v32 + 16);
        }
        while ( v34 );
LABEL_51:
        if ( v33 == &v56 || v47 < (unsigned int)v33[8] )
        {
LABEL_53:
          v49 = v47;
          v48 = 0x20000000;
          v50 = 0;
          v51 = 0;
          v52 = 0;
          sub_1E1AFD0(v45, &v48);
          *(_QWORD *)(a1[46] + 8LL * v47) = v45;
          v37 = a1[45];
          if ( !v37 )
            BUG();
          v38 = (unsigned __int16 *)(*(_QWORD *)(v37 + 56)
                                   + 2LL * *(unsigned int *)(*(_QWORD *)(v37 + 8) + 24LL * v47 + 4));
          v39 = *v38;
          v40 = v38 + 1;
          v41 = v47 + (_DWORD)v39;
          if ( !(_WORD)v39 )
            v40 = 0;
          while ( v40 )
          {
            while ( 1 )
            {
              ++v40;
              LODWORD(v48) = (unsigned __int16)v41;
              sub_1DCB780((__int64)&v61, (unsigned int *)&v48, v37, v39, v36);
              v42 = *(v40 - 1);
              v37 = (unsigned int)(v42 + v41);
              if ( !(_WORD)v42 )
                break;
              v41 += v42;
              if ( !v40 )
                goto LABEL_28;
            }
            v40 = 0;
          }
        }
      }
      else
      {
        v24 = v53;
        v25 = &v53[4 * (unsigned int)v54];
        if ( v53 == v25 )
          goto LABEL_53;
        while ( v47 != *v24 )
        {
          if ( v25 == (_BYTE *)++v24 )
            goto LABEL_53;
        }
        if ( v24 == (_DWORD *)v25 )
          goto LABEL_53;
      }
LABEL_28:
      v26 = *v21++;
      if ( v26 )
        v47 += v26;
      else
        v21 = 0;
    }
  }
LABEL_3:
  v11 = v6[45];
  if ( !v11 )
    BUG();
  v12 = *(_QWORD *)(v11 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v11 + 8) + 24 * v5 + 4);
  while ( 1 )
  {
    result = v12;
    if ( !v12 )
      break;
    while ( 1 )
    {
      result += 2;
      *(_QWORD *)(v6[49] + 8LL * v8) = a3;
      v14 = *(_WORD *)(result - 2);
      v12 = 0;
      if ( !v14 )
        break;
      v8 += v14;
      if ( !result )
        return result;
    }
  }
  return result;
}
