// Function: sub_2AD3030
// Address: 0x2ad3030
//
__int64 *__fastcall sub_2AD3030(__int64 a1, __int64 a2)
{
  __int64 *result; // rax
  int v3; // ecx
  int v4; // r10d
  int v5; // ecx
  unsigned int i; // edx
  __int64 v7; // r8
  unsigned int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rbx
  __int64 v12; // r14
  unsigned __int8 *v13; // r12
  __int64 v14; // r12
  _QWORD *v15; // rax
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // r12
  __int64 v19; // rdx
  __int64 v20; // r13
  _QWORD *v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rdx
  signed __int64 v24; // rax
  int v25; // edx
  bool v26; // sf
  __int64 v27; // r13
  __int64 v28; // r12
  __int64 v29; // rbx
  _QWORD **v30; // r14
  _QWORD **v31; // r13
  _QWORD *v32; // rbx
  _QWORD *v33; // rdx
  _OWORD *v34; // rax
  _QWORD *v35; // rax
  _QWORD *v36; // rax
  __int64 v37; // rdx
  _QWORD *v38; // [rsp+8h] [rbp-128h]
  __int64 v39; // [rsp+18h] [rbp-118h]
  __int64 v40; // [rsp+20h] [rbp-110h]
  __int64 v41; // [rsp+28h] [rbp-108h]
  __int64 v42; // [rsp+30h] [rbp-100h]
  _QWORD **v43; // [rsp+30h] [rbp-100h]
  __int64 v44; // [rsp+38h] [rbp-F8h]
  __int64 *v45; // [rsp+40h] [rbp-F0h]
  __int64 v46; // [rsp+50h] [rbp-E0h]
  __int64 v47; // [rsp+58h] [rbp-D8h]
  __int64 *v48; // [rsp+68h] [rbp-C8h]
  __int64 j; // [rsp+70h] [rbp-C0h]
  __int64 v50[2]; // [rsp+78h] [rbp-B8h] BYREF
  _QWORD *v51; // [rsp+88h] [rbp-A8h] BYREF
  _QWORD *v52; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v53; // [rsp+98h] [rbp-98h]
  __int64 v54; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v55; // [rsp+A8h] [rbp-88h]
  __int64 v56; // [rsp+B0h] [rbp-80h]
  unsigned int v57; // [rsp+B8h] [rbp-78h]
  __int64 v58; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v59; // [rsp+D0h] [rbp-60h]
  __int64 v60; // [rsp+D8h] [rbp-58h]
  _QWORD *v61; // [rsp+E0h] [rbp-50h] BYREF
  __int64 v62; // [rsp+E8h] [rbp-48h]
  _QWORD **v63; // [rsp+F0h] [rbp-40h]
  _QWORD **v64; // [rsp+F8h] [rbp-38h]

  result = (__int64 *)(unsigned int)a2;
  v50[0] = a2;
  if ( (BYTE4(a2) || (_DWORD)a2 != 1) && (_DWORD)a2 )
  {
    v3 = *(_DWORD *)(a1 + 152);
    if ( v3 )
    {
      v4 = 1;
      v5 = v3 - 1;
      for ( i = v5 & ((BYTE4(a2) == 0) + 37 * a2 - 1); ; i = v5 & v8 )
      {
        v7 = *(_QWORD *)(a1 + 136) + 40LL * i;
        if ( *(_DWORD *)v7 == (_DWORD)a2 && BYTE4(a2) == *(_BYTE *)(v7 + 4) )
          break;
        if ( *(_DWORD *)v7 == -1 && *(_BYTE *)(v7 + 4) )
          goto LABEL_13;
        v8 = v4 + i;
        ++v4;
      }
    }
    else
    {
LABEL_13:
      v44 = sub_2ACFA80(a1 + 128, (__int64)v50);
      v46 = a1 + 64;
      v9 = sub_2ACFE10(a1 + 64, (__int64)v50);
      sub_270F0C0(v9, (__int64)v50);
      v10 = *(_QWORD *)(a1 + 416);
      result = *(__int64 **)(v10 + 32);
      v45 = *(__int64 **)(v10 + 40);
      if ( v45 != result )
      {
        v48 = *(__int64 **)(v10 + 32);
        v41 = a1 + 384;
        do
        {
          v11 = *v48;
          if ( *(_BYTE *)(a1 + 108) && *(_DWORD *)(a1 + 100)
            || (unsigned __int8)sub_31A6C30(*(_QWORD *)(a1 + 440), *v48) )
          {
            v12 = *(_QWORD *)(v11 + 56);
            for ( j = v11 + 48; j != v12; v12 = *(_QWORD *)(v12 + 8) )
            {
              v13 = (unsigned __int8 *)(v12 - 24);
              if ( !v12 )
                v13 = 0;
              if ( sub_2AC3650(a1, v13, v50[0]) )
              {
                v54 = 0;
                v55 = 0;
                v56 = 0;
                v57 = 0;
                if ( !(unsigned __int8)sub_2AB2DA0(a1, (__int64)v13, v50[0])
                  && !BYTE4(v50[0])
                  && !sub_2AB4AE0((_DWORD *)a1, v13) )
                {
                  v24 = sub_2AD24E0(a1, (__int64)v13, (__int64)&v54, v50[0]);
                  v26 = v25 < 0;
                  if ( !v25 )
                    v26 = v24 < 0;
                  if ( !v26 )
                  {
                    v42 = v55 + 24LL * v57;
                    sub_2ABD9E0(&v58, &v54);
                    v27 = v59;
                    if ( v59 != v42 )
                    {
                      v28 = v11;
                      v29 = v60;
                      do
                      {
                        if ( !(unsigned __int8)sub_2AC1850(v44, (__int64 *)v27, &v61) )
                        {
                          v35 = sub_2AD00C0(v44, (__int64 *)v27, v61);
                          *v35 = *(_QWORD *)v27;
                          *(__m128i *)(v35 + 1) = _mm_loadu_si128((const __m128i *)(v27 + 8));
                        }
                        do
                          v27 += 24;
                        while ( v29 != v27 && (*(_QWORD *)v27 == -4096 || *(_QWORD *)v27 == -8192) );
                      }
                      while ( v42 != v27 );
                      v11 = v28;
                    }
                    sub_2ABD9E0(&v61, &v54);
                    v43 = (_QWORD **)(v55 + 24LL * v57);
                    if ( v63 != v43 )
                    {
                      v40 = v11;
                      v39 = v12;
                      v30 = v63;
                      v31 = v64;
                      do
                      {
                        v32 = *v30;
                        if ( *(_BYTE *)*v30 == 85 )
                        {
                          v52 = *v30;
                          v53 = v50[0];
                          if ( sub_2AC3590(v41, (__int64 *)&v52) )
                          {
                            *(_DWORD *)sub_2AC7E90(v41, (__int64)&v52) = 5;
                            v51 = v32;
                            if ( (unsigned __int8)sub_2AC1850((__int64)&v54, (__int64 *)&v51, &v52) )
                            {
                              v33 = v52 + 1;
                            }
                            else
                            {
                              v36 = sub_2AD00C0((__int64)&v54, (__int64 *)&v51, v52);
                              v37 = (__int64)v51;
                              v36[1] = 0;
                              *v36 = v37;
                              v33 = v36 + 1;
                              *((_DWORD *)v36 + 4) = 0;
                            }
                            v38 = v33;
                            v52 = v32;
                            v53 = v50[0];
                            v34 = sub_2AC7E90(v41, (__int64)&v52);
                            *((_QWORD *)v34 + 4) = *v38;
                            *((_DWORD *)v34 + 10) = *((_DWORD *)v38 + 2);
                          }
                        }
                        for ( v30 += 3; v31 != v30; v30 += 3 )
                        {
                          if ( *v30 != (_QWORD *)-8192LL && *v30 != (_QWORD *)-4096LL )
                            break;
                        }
                      }
                      while ( v30 != v43 );
                      v11 = v40;
                      v12 = v39;
                    }
                  }
                }
                v14 = sub_2ACFE10(v46, (__int64)v50);
                v15 = sub_AE6EC0(v14, v11);
                if ( *(_BYTE *)(v14 + 28) )
                  v16 = *(unsigned int *)(v14 + 20);
                else
                  v16 = *(unsigned int *)(v14 + 16);
                v17 = *(_QWORD *)(v14 + 8) + 8 * v16;
                v61 = v15;
                v62 = v17;
                sub_254BBF0((__int64)&v61);
                v18 = *(_QWORD *)(v11 + 16);
                if ( v18 )
                {
                  while ( 1 )
                  {
                    v19 = *(_QWORD *)(v18 + 24);
                    if ( (unsigned __int8)(*(_BYTE *)v19 - 30) <= 0xAu )
                      break;
                    v18 = *(_QWORD *)(v18 + 8);
                    if ( !v18 )
                      goto LABEL_33;
                  }
LABEL_31:
                  v20 = *(_QWORD *)(v19 + 40);
                  if ( v11 == sub_AA56F0(v20) )
                  {
                    v47 = sub_2ACFE10(v46, (__int64)v50);
                    v21 = sub_AE6EC0(v47, v20);
                    if ( *(_BYTE *)(v47 + 28) )
                      v22 = *(unsigned int *)(v47 + 20);
                    else
                      v22 = *(unsigned int *)(v47 + 16);
                    v23 = *(_QWORD *)(v47 + 8) + 8 * v22;
                    v61 = v21;
                    v62 = v23;
                    sub_254BBF0((__int64)&v61);
                  }
                  while ( 1 )
                  {
                    v18 = *(_QWORD *)(v18 + 8);
                    if ( !v18 )
                      break;
                    v19 = *(_QWORD *)(v18 + 24);
                    if ( (unsigned __int8)(*(_BYTE *)v19 - 30) <= 0xAu )
                      goto LABEL_31;
                  }
                }
LABEL_33:
                sub_C7D6A0(v55, 24LL * v57, 8);
              }
            }
          }
          result = ++v48;
        }
        while ( v45 != v48 );
      }
    }
  }
  return result;
}
