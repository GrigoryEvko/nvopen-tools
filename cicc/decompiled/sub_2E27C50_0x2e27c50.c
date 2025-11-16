// Function: sub_2E27C50
// Address: 0x2e27c50
//
__int64 __fastcall sub_2E27C50(_QWORD *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r10
  _QWORD *v6; // r12
  unsigned int v7; // r9d
  __int64 v8; // r11
  unsigned __int16 v9; // bx
  __int64 v10; // r14
  _QWORD *v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 result; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  unsigned int v20; // r9d
  __int64 v21; // r10
  __int64 v22; // r11
  __int64 v23; // r14
  __int64 v24; // r8
  __int64 v25; // r11
  __int64 v26; // r9
  __int64 v27; // r10
  __int64 v28; // rax
  __int16 *v29; // rax
  __int16 *v30; // rdx
  int v31; // eax
  __int64 v32; // rcx
  __int64 v33; // r13
  _WORD *v34; // rax
  unsigned __int64 v35; // rdx
  unsigned __int64 v36; // rax
  int *v37; // rsi
  _DWORD *v38; // rax
  __int16 *v39; // rax
  __int64 v40; // rcx
  int v41; // ebx
  unsigned __int16 v42; // ax
  __int64 v43; // r12
  unsigned int v44; // eax
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  unsigned __int64 v48; // rax
  int *v49; // rsi
  __int64 v50; // [rsp+18h] [rbp-168h]
  __int64 v51; // [rsp+28h] [rbp-158h]
  unsigned __int16 v52; // [rsp+30h] [rbp-150h]
  __int64 v54; // [rsp+48h] [rbp-138h]
  __int64 v55; // [rsp+48h] [rbp-138h]
  __int64 v56; // [rsp+48h] [rbp-138h]
  unsigned int v58; // [rsp+50h] [rbp-130h]
  __int64 v59; // [rsp+50h] [rbp-130h]
  __int64 v60; // [rsp+50h] [rbp-130h]
  __int16 *v62; // [rsp+58h] [rbp-128h]
  unsigned int v63; // [rsp+58h] [rbp-128h]
  unsigned int v64; // [rsp+58h] [rbp-128h]
  unsigned __int16 v65; // [rsp+6Eh] [rbp-112h] BYREF
  __int64 v66; // [rsp+70h] [rbp-110h] BYREF
  int v67; // [rsp+78h] [rbp-108h]
  __int64 v68; // [rsp+80h] [rbp-100h]
  __int64 v69; // [rsp+88h] [rbp-F8h]
  __int64 v70; // [rsp+90h] [rbp-F0h]
  _BYTE *v71; // [rsp+A0h] [rbp-E0h] BYREF
  __int64 v72; // [rsp+A8h] [rbp-D8h]
  _BYTE v73[24]; // [rsp+B0h] [rbp-D0h] BYREF
  int v74; // [rsp+C8h] [rbp-B8h] BYREF
  unsigned __int64 v75; // [rsp+D0h] [rbp-B0h]
  int *v76; // [rsp+D8h] [rbp-A8h]
  int *v77; // [rsp+E0h] [rbp-A0h]
  __int64 v78; // [rsp+E8h] [rbp-98h]
  unsigned __int64 v79; // [rsp+F0h] [rbp-90h] BYREF
  __int64 v80; // [rsp+F8h] [rbp-88h]
  __int64 v81; // [rsp+100h] [rbp-80h]
  __int64 v82; // [rsp+108h] [rbp-78h] BYREF
  __int64 v83; // [rsp+110h] [rbp-70h]
  int v84; // [rsp+120h] [rbp-60h] BYREF
  unsigned __int64 v85; // [rsp+128h] [rbp-58h]
  int *v86; // [rsp+130h] [rbp-50h]
  int *v87; // [rsp+138h] [rbp-48h]
  __int64 v88; // [rsp+140h] [rbp-40h]

  v5 = a3;
  v6 = a1;
  v7 = a2;
  v8 = 24LL * a2;
  v9 = a2;
  v10 = *(_QWORD *)(a1[13] + 8LL * a2);
  v11 = (_QWORD *)(a1[16] + 8LL * a2);
  if ( v10 )
  {
    if ( !*v11 )
    {
      v44 = sub_2E8E710(v10, a2, 0, 0, 0);
      v7 = a2;
      v5 = a3;
      v8 = 24LL * a2;
      if ( v44 == -1 || (v45 = 5LL * v44, !(*(_QWORD *)(v10 + 32) + 40LL * v44)) )
      {
        v79 = 805306368;
        v81 = 0;
        LODWORD(v80) = a2;
        v82 = 0;
        v83 = 0;
        sub_2E8F270(v10, &v79, v45, v46, v47, a2);
        v8 = 24LL * a2;
        v5 = a3;
        v7 = a2;
      }
    }
  }
  else if ( !*v11 )
  {
    v71 = v73;
    v72 = 0x400000000LL;
    v74 = 0;
    v75 = 0;
    v76 = &v74;
    v77 = &v74;
    v78 = 0;
    v16 = sub_2E271D0((__int64)a1, a2, (__int64)&v71, a4, a5, a2);
    v20 = a2;
    v21 = a3;
    v50 = v16;
    v22 = 24LL * a2;
    v23 = v16;
    if ( v16 )
    {
      v79 = 805306368;
      v81 = 0;
      LODWORD(v80) = a2;
      v82 = 0;
      v83 = 0;
      sub_2E8F270(v16, &v79, v17, v18, v19, a2);
      v25 = 24LL * a2;
      v26 = a2;
      v27 = a3;
      *(_QWORD *)(a1[13] + 8LL * a2) = v23;
      v79 = (unsigned __int64)&v82;
      v86 = &v84;
      v87 = &v84;
      v28 = a1[12];
      v80 = 0;
      v81 = 8;
      v84 = 0;
      v85 = 0;
      v88 = 0;
      v54 = v25;
      v29 = (__int16 *)(*(_QWORD *)(v28 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v28 + 8) + v25 + 4));
      v30 = v29 + 1;
      v31 = *v29;
      v32 = a2 + v31;
      v58 = a2 + v31;
      if ( (_WORD)v31 )
      {
        v62 = v30;
        v33 = (unsigned __int16)(a2 + v31);
        v52 = a2;
        v51 = v27;
LABEL_11:
        v34 = (_WORD *)v79;
        v35 = v79 + 2 * v80;
        if ( v79 != v35 )
        {
          while ( *v34 != (_WORD)v33 )
          {
            if ( (_WORD *)v35 == ++v34 )
              goto LABEL_25;
          }
          if ( (_WORD *)v35 != v34 )
            goto LABEL_16;
        }
LABEL_25:
        if ( v78 )
        {
          v48 = v75;
          if ( v75 )
          {
            v49 = &v74;
            do
            {
              v32 = *(_QWORD *)(v48 + 16);
              v35 = *(_QWORD *)(v48 + 24);
              if ( (unsigned int)(unsigned __int16)v33 > *(_DWORD *)(v48 + 32) )
              {
                v48 = *(_QWORD *)(v48 + 24);
              }
              else
              {
                v49 = (int *)v48;
                v48 = *(_QWORD *)(v48 + 16);
              }
            }
            while ( v48 );
            if ( v49 != &v74 && (unsigned __int16)v33 >= (unsigned int)v49[8] )
              goto LABEL_16;
          }
        }
        else
        {
          v38 = v71;
          v35 = (unsigned __int64)&v71[4 * (unsigned int)v72];
          if ( v71 != (_BYTE *)v35 )
          {
            while ( (unsigned __int16)v33 != *v38 )
            {
              if ( (_DWORD *)v35 == ++v38 )
                goto LABEL_31;
            }
            if ( v38 != (_DWORD *)v35 )
              goto LABEL_16;
          }
        }
LABEL_31:
        v67 = (unsigned __int16)v33;
        v66 = 0x20000000;
        v68 = 0;
        v69 = 0;
        v70 = 0;
        sub_2E8F270(v50, &v66, v35, v32, v24, v26);
        *(_QWORD *)(a1[13] + 8 * v33) = v50;
        v39 = (__int16 *)(*(_QWORD *)(a1[12] + 56LL) + 2LL * *(unsigned int *)(*(_QWORD *)(a1[12] + 8LL) + 24 * v33 + 4));
        v35 = (unsigned int)*v39;
        v40 = (__int64)(v39 + 1);
        v41 = v35 + (unsigned __int16)v33;
        if ( *v39 )
        {
          v42 = *v39 + v33;
          v43 = v40;
          while ( 1 )
          {
            v43 += 2;
            v65 = v42;
            sub_2E27A70((__int64)&v66, (__int64)&v79, &v65, v40, v24);
            if ( !*(_WORD *)(v43 - 2) )
              break;
            v41 += *(__int16 *)(v43 - 2);
            v42 = v41;
          }
        }
LABEL_16:
        while ( *v62++ )
        {
          v58 += *(v62 - 1);
          v32 = v58;
          v33 = (unsigned __int16)v58;
          if ( !v88 )
            goto LABEL_11;
          v36 = v85;
          if ( v85 )
          {
            v37 = &v84;
            do
            {
              while ( 1 )
              {
                v32 = *(_QWORD *)(v36 + 16);
                v35 = *(_QWORD *)(v36 + 24);
                if ( *(_WORD *)(v36 + 32) >= (unsigned __int16)v58 )
                  break;
                v36 = *(_QWORD *)(v36 + 24);
                if ( !v35 )
                  goto LABEL_23;
              }
              v37 = (int *)v36;
              v36 = *(_QWORD *)(v36 + 16);
            }
            while ( v32 );
LABEL_23:
            if ( v37 != &v84 && *((_WORD *)v37 + 16) <= (unsigned __int16)v58 )
              continue;
          }
          goto LABEL_25;
        }
        LODWORD(v26) = a2;
        v6 = a1;
        v9 = v52;
        v27 = v51;
        v25 = v54;
      }
      v55 = v25;
      v59 = v27;
      v63 = v26;
      sub_2E24350(v85);
      v20 = v63;
      v21 = v59;
      v22 = v55;
      if ( (__int64 *)v79 != &v82 )
      {
        _libc_free(v79);
        v22 = v55;
        v21 = v59;
        v20 = v63;
      }
    }
    v56 = v22;
    v60 = v21;
    v64 = v20;
    sub_2E24520(v75);
    v7 = v64;
    v5 = v60;
    v8 = v56;
    if ( v71 != v73 )
    {
      _libc_free((unsigned __int64)v71);
      v7 = v64;
      v5 = v60;
      v8 = v56;
    }
  }
  v12 = v6[12];
  v13 = *(_QWORD *)(v12 + 8);
  result = *(_QWORD *)(v12 + 56);
  v15 = result + 2LL * *(unsigned int *)(v13 + v8 + 4);
  if ( v15 )
  {
    while ( 1 )
    {
      v15 += 2;
      *(_QWORD *)(v6[16] + 8LL * v9) = v5;
      result = (unsigned int)*(__int16 *)(v15 - 2);
      if ( !*(_WORD *)(v15 - 2) )
        break;
      v7 += result;
      v9 = v7;
    }
  }
  return result;
}
