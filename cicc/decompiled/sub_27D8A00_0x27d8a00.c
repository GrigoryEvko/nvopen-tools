// Function: sub_27D8A00
// Address: 0x27d8a00
//
__int64 __fastcall sub_27D8A00(__int64 a1, __int64 a2)
{
  __int64 *v2; // r14
  __int64 *v3; // r13
  __int64 v4; // r15
  __int64 v5; // rdx
  __int64 v6; // rbx
  __int64 v7; // rcx
  unsigned int v8; // eax
  __int64 v9; // r12
  __int64 v10; // r15
  __int64 v11; // rbx
  __int64 v12; // rdx
  _QWORD *v13; // rax
  _QWORD *v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  char v19; // r12
  __int64 *v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rbx
  _QWORD *v25; // r12
  __int64 v26; // rax
  unsigned int v27; // eax
  __int64 v28; // rdx
  int v29; // edx
  int v30; // eax
  __int64 *v31; // rcx
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 *v34; // r9
  unsigned __int64 v35; // rsi
  int v36; // eax
  unsigned __int64 *v37; // rdi
  __int64 v38; // rax
  __int64 v39; // r12
  char v40; // r10
  __int64 v41; // rsi
  __int64 *v42; // rax
  __int64 v43; // r8
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 *v46; // rsi
  unsigned __int64 v47; // r9
  int v48; // eax
  unsigned __int64 *v49; // rdi
  __int64 v50; // rax
  char *v51; // rbx
  char *v52; // r12
  __int64 v55; // [rsp+8h] [rbp-228h]
  __int64 v56; // [rsp+20h] [rbp-210h]
  __int64 v57; // [rsp+28h] [rbp-208h]
  unsigned __int8 v58; // [rsp+37h] [rbp-1F9h]
  const __m128i *v59; // [rsp+40h] [rbp-1F0h]
  __int64 v60; // [rsp+48h] [rbp-1E8h]
  __int64 v61; // [rsp+50h] [rbp-1E0h] BYREF
  __int64 v62; // [rsp+58h] [rbp-1D8h]
  char *v63; // [rsp+60h] [rbp-1D0h]
  __int64 v64; // [rsp+70h] [rbp-1C0h] BYREF
  char *v65; // [rsp+78h] [rbp-1B8h]
  __int64 v66; // [rsp+80h] [rbp-1B0h]
  int v67; // [rsp+88h] [rbp-1A8h]
  char v68; // [rsp+8Ch] [rbp-1A4h]
  char v69; // [rsp+90h] [rbp-1A0h] BYREF
  __int64 v70; // [rsp+D0h] [rbp-160h] BYREF
  char *v71; // [rsp+D8h] [rbp-158h]
  __int64 v72; // [rsp+E0h] [rbp-150h]
  int v73; // [rsp+E8h] [rbp-148h]
  char v74; // [rsp+ECh] [rbp-144h]
  char v75; // [rsp+F0h] [rbp-140h] BYREF
  __int64 v76; // [rsp+130h] [rbp-100h] BYREF
  __int64 v77; // [rsp+138h] [rbp-F8h]
  _BYTE v78[240]; // [rsp+140h] [rbp-F0h] BYREF

  v2 = &v64;
  v3 = &v70;
  v65 = &v69;
  v71 = &v75;
  v56 = a1 + 72;
  v59 = (const __m128i *)a2;
  v64 = 0;
  v66 = 8;
  v67 = 0;
  v68 = 1;
  v70 = 0;
  v72 = 8;
  v73 = 0;
  v74 = 1;
  v58 = 0;
  while ( 1 )
  {
    v4 = *(_QWORD *)(a1 + 80);
    if ( v4 != v56 )
    {
      while ( 1 )
      {
        v5 = v59[1].m128i_i64[1];
        if ( v4 )
        {
          v6 = v4 - 24;
          v7 = (unsigned int)(*(_DWORD *)(v4 + 20) + 1);
          v8 = *(_DWORD *)(v4 + 20) + 1;
        }
        else
        {
          v6 = 0;
          v7 = 0;
          v8 = 0;
        }
        if ( v8 < *(_DWORD *)(v5 + 32) && *(_QWORD *)(*(_QWORD *)(v5 + 24) + 8 * v7) )
          break;
LABEL_32:
        v4 = *(_QWORD *)(v4 + 8);
        if ( v4 == v56 )
          goto LABEL_33;
      }
      v9 = *(_QWORD *)(v6 + 56);
      v76 = (__int64)v78;
      v77 = 0x800000000LL;
      v60 = v6 + 48;
      if ( v9 == v6 + 48 )
      {
LABEL_22:
        v63 = 0;
        a2 = v59->m128i_i64[1];
        sub_F5C330((__int64)&v76, (__int64 *)a2, 0, (__int64)&v61);
        if ( v63 )
        {
          a2 = (__int64)&v61;
          ((void (__fastcall *)(__int64 *, __int64 *, __int64))v63)(&v61, &v61, 3);
        }
        v24 = v76;
        v25 = (_QWORD *)(v76 + 24LL * (unsigned int)v77);
        if ( (_QWORD *)v76 != v25 )
        {
          do
          {
            v26 = *(v25 - 1);
            v25 -= 3;
            if ( v26 != -4096 && v26 != 0 && v26 != -8192 )
              sub_BD60C0(v25);
          }
          while ( (_QWORD *)v24 != v25 );
          v25 = (_QWORD *)v76;
        }
        if ( v25 != (_QWORD *)v78 )
          _libc_free((unsigned __int64)v25);
        goto LABEL_32;
      }
      v55 = v4;
      v10 = v9;
      while ( 1 )
      {
        v11 = 0;
        v12 = *((unsigned int *)v2 + 5);
        if ( v10 )
          v11 = v10 - 24;
        if ( (_DWORD)v12 != *((_DWORD *)v2 + 6) )
        {
          if ( *((_BYTE *)v2 + 28) )
          {
            v13 = (_QWORD *)v2[1];
            v14 = &v13[v12];
            if ( v13 == v14 )
              goto LABEL_20;
            while ( v11 != *v13 )
            {
              if ( v14 == ++v13 )
                goto LABEL_20;
            }
          }
          else if ( !sub_C8CA60((__int64)v2, v11) )
          {
            goto LABEL_20;
          }
        }
        v19 = sub_F50EE0((unsigned __int8 *)v11, 0);
        if ( v19 )
        {
          v63 = (char *)v11;
          v61 = 6;
          v62 = 0;
          if ( v11 != 0 && v11 != -4096 && v11 != -8192 )
            sub_BD73F0((__int64)&v61);
          v32 = (unsigned int)v77;
          v33 = v76;
          v34 = &v61;
          v35 = (unsigned int)v77 + 1LL;
          v36 = v77;
          if ( v35 > HIDWORD(v77) )
          {
            if ( v76 > (unsigned __int64)&v61
              || (unsigned __int64)&v61 >= v76 + 24 * (unsigned __int64)(unsigned int)v77 )
            {
              sub_F39130((__int64)&v76, v35, (unsigned int)v77, v76, v17, (__int64)&v61);
              v32 = (unsigned int)v77;
              v33 = v76;
              v34 = &v61;
              v36 = v77;
            }
            else
            {
              v51 = (char *)&v61 - v76;
              sub_F39130((__int64)&v76, v35, (unsigned int)v77, v76, v17, (__int64)&v61);
              v33 = v76;
              v32 = (unsigned int)v77;
              v34 = (__int64 *)&v51[v76];
              v36 = v77;
            }
          }
          v37 = (unsigned __int64 *)(v33 + 24 * v32);
          if ( v37 )
          {
            *v37 = 6;
            v38 = v34[2];
            v37[1] = 0;
            v37[2] = v38;
            if ( v38 != 0 && v38 != -4096 && v38 != -8192 )
              sub_BD6050(v37, *v34 & 0xFFFFFFFFFFFFFFF8LL);
            v36 = v77;
          }
          LODWORD(v77) = v36 + 1;
          if ( v63 != 0 && v63 + 4096 != 0 && v63 != (char *)-8192LL )
            sub_BD60C0(&v61);
          v58 = v19;
          goto LABEL_20;
        }
        if ( *(_QWORD *)(v11 + 16) )
        {
          v57 = sub_1020E10(v11, v59, v15, v16, v17, v18);
          if ( v57 )
          {
            v39 = *(_QWORD *)(v11 + 16);
            if ( !v39 )
            {
LABEL_64:
              sub_BD84D0(v11, v57);
              v58 = sub_F50EE0((unsigned __int8 *)v11, 0);
              if ( v58 )
              {
                v61 = 6;
                v62 = 0;
                v63 = (char *)v11;
                if ( v11 != -8192 && v11 != -4096 )
                  sub_BD73F0((__int64)&v61);
                v44 = (unsigned int)v77;
                v45 = v76;
                v46 = &v61;
                v47 = (unsigned int)v77 + 1LL;
                v48 = v77;
                if ( v47 > HIDWORD(v77) )
                {
                  if ( v76 > (unsigned __int64)&v61
                    || (unsigned __int64)&v61 >= v76 + 24 * (unsigned __int64)(unsigned int)v77 )
                  {
                    sub_F39130((__int64)&v76, (unsigned int)v77 + 1LL, (unsigned int)v77, v76, v43, v47);
                    v44 = (unsigned int)v77;
                    v45 = v76;
                    v46 = &v61;
                    v48 = v77;
                  }
                  else
                  {
                    v52 = (char *)&v61 - v76;
                    sub_F39130((__int64)&v76, (unsigned int)v77 + 1LL, (unsigned int)v77, v76, v43, v47);
                    v45 = v76;
                    v44 = (unsigned int)v77;
                    v46 = (__int64 *)&v52[v76];
                    v48 = v77;
                  }
                }
                v49 = (unsigned __int64 *)(v45 + 24 * v44);
                if ( v49 )
                {
                  *v49 = 6;
                  v50 = v46[2];
                  v49[1] = 0;
                  v49[2] = v50;
                  if ( v50 != -4096 && v50 != 0 && v50 != -8192 )
                    sub_BD6050(v49, *v46 & 0xFFFFFFFFFFFFFFF8LL);
                  v48 = v77;
                }
                LODWORD(v77) = v48 + 1;
                if ( v63 + 4096 != 0 && v63 != 0 && v63 != (char *)-8192LL )
                  sub_BD60C0(&v61);
              }
              else
              {
                v58 = 1;
              }
              goto LABEL_20;
            }
            v40 = *((_BYTE *)v3 + 28);
            while ( 2 )
            {
              while ( 1 )
              {
                v41 = *(_QWORD *)(v39 + 24);
                if ( !v40 )
                  break;
                v42 = (__int64 *)v3[1];
                v21 = *((unsigned int *)v3 + 5);
                v20 = &v42[v21];
                if ( v42 == v20 )
                {
LABEL_68:
                  if ( (unsigned int)v21 >= *((_DWORD *)v3 + 4) )
                    break;
                  v21 = (unsigned int)(v21 + 1);
                  *((_DWORD *)v3 + 5) = v21;
                  *v20 = v41;
                  v40 = *((_BYTE *)v3 + 28);
                  ++*v3;
                  v39 = *(_QWORD *)(v39 + 8);
                  if ( !v39 )
                    goto LABEL_64;
                }
                else
                {
                  while ( v41 != *v42 )
                  {
                    if ( v20 == ++v42 )
                      goto LABEL_68;
                  }
                  v39 = *(_QWORD *)(v39 + 8);
                  if ( !v39 )
                    goto LABEL_64;
                }
              }
              sub_C8CC70((__int64)v3, v41, (__int64)v20, v21, v22, v23);
              v39 = *(_QWORD *)(v39 + 8);
              v40 = *((_BYTE *)v3 + 28);
              if ( !v39 )
                goto LABEL_64;
              continue;
            }
          }
        }
LABEL_20:
        v10 = *(_QWORD *)(v10 + 8);
        if ( v60 == v10 )
        {
          v4 = v55;
          goto LABEL_22;
        }
      }
    }
LABEL_33:
    ++*v2;
    if ( *((_BYTE *)v2 + 28) )
      goto LABEL_38;
    v27 = 4 * (*((_DWORD *)v2 + 5) - *((_DWORD *)v2 + 6));
    v28 = *((unsigned int *)v2 + 4);
    if ( v27 < 0x20 )
      v27 = 32;
    if ( (unsigned int)v28 <= v27 )
    {
      a2 = 0xFFFFFFFFLL;
      memset((void *)v2[1], -1, 8 * v28);
LABEL_38:
      *(__int64 *)((char *)v2 + 20) = 0;
      goto LABEL_39;
    }
    sub_C8C990((__int64)v2, a2);
LABEL_39:
    v29 = *((_DWORD *)v3 + 5);
    v30 = *((_DWORD *)v3 + 6);
    v31 = v3;
    v3 = v2;
    if ( v29 == v30 )
      break;
    v2 = v31;
  }
  if ( !v74 )
  {
    _libc_free((unsigned __int64)v71);
    if ( v68 )
      return v58;
LABEL_97:
    _libc_free((unsigned __int64)v65);
    return v58;
  }
  if ( !v68 )
    goto LABEL_97;
  return v58;
}
