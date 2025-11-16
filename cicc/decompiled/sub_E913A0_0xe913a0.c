// Function: sub_E913A0
// Address: 0xe913a0
//
__int64 __fastcall sub_E913A0(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // edx
  _QWORD *v7; // r15
  unsigned __int64 v9; // r8
  __int64 v10; // rdi
  _QWORD *v11; // rbx
  __int64 v12; // rax
  _QWORD *v13; // r12
  char *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rax
  unsigned __int64 v19; // rsi
  _BYTE *v20; // rax
  _BYTE *v21; // rdx
  char *v22; // r12
  __int64 v23; // rdi
  unsigned __int64 v24; // rax
  __int64 v25; // rdx
  _QWORD *v26; // r12
  __int64 v27; // rbx
  __int64 v28; // r13
  _QWORD *v29; // rax
  __int64 result; // rax
  unsigned __int64 v31; // rdx
  __int64 v32; // r14
  char *v33; // rax
  char *v34; // rbx
  char *v35; // rdi
  _BYTE *v36; // rax
  __int64 *v37; // rbx
  __int64 v38; // r13
  __int64 v39; // r14
  unsigned __int64 v40; // rax
  _QWORD *v41; // rsi
  size_t v42; // rdx
  int *v43; // rsi
  __int64 v44; // [rsp+10h] [rbp-170h]
  __int64 *v45; // [rsp+18h] [rbp-168h]
  __int64 v46; // [rsp+20h] [rbp-160h]
  _BYTE *v47; // [rsp+28h] [rbp-158h]
  size_t v48; // [rsp+38h] [rbp-148h]
  unsigned int *v49[2]; // [rsp+40h] [rbp-140h] BYREF
  __int64 v50; // [rsp+50h] [rbp-130h] BYREF
  _BYTE *v51; // [rsp+58h] [rbp-128h]
  _BYTE *v52; // [rsp+60h] [rbp-120h]
  char *v53; // [rsp+70h] [rbp-110h] BYREF
  __int64 v54; // [rsp+78h] [rbp-108h]
  _BYTE v55[48]; // [rsp+80h] [rbp-100h] BYREF
  unsigned __int64 v56[26]; // [rsp+B0h] [rbp-D0h] BYREF

  v6 = 0;
  v7 = a2;
  v9 = *(_QWORD *)(a1 + 24);
  v10 = 0;
  v44 = *((_QWORD *)a2 + 1);
  v53 = v55;
  v54 = 0x300000000LL;
  if ( v9 > 3 )
  {
    a2 = v55;
    sub_C8D5F0((__int64)&v53, v55, v9, 0x10u, v9, a6);
    v10 = (unsigned int)v54;
    v6 = v54;
  }
  v11 = *(_QWORD **)(a1 + 16);
  if ( v11 )
  {
    do
    {
      v12 = v6;
      v13 = v11 + 2;
      if ( v6 >= (unsigned __int64)HIDWORD(v54) )
      {
        v31 = v6 + 1LL;
        v32 = v11[1];
        if ( HIDWORD(v54) < (unsigned __int64)(v12 + 1) )
        {
          a2 = v55;
          sub_C8D5F0((__int64)&v53, v55, v31, 0x10u, v9, a6);
          v12 = (unsigned int)v54;
        }
        v33 = &v53[16 * v12];
        *(_QWORD *)v33 = v32;
        *((_QWORD *)v33 + 1) = v13;
        v6 = v54 + 1;
        LODWORD(v54) = v54 + 1;
      }
      else
      {
        v14 = &v53[16 * v6];
        if ( v14 )
        {
          v15 = v11[1];
          *((_QWORD *)v14 + 1) = v13;
          *(_QWORD *)v14 = v15;
          v6 = v54;
        }
        LODWORD(v54) = ++v6;
      }
      v11 = (_QWORD *)*v11;
    }
    while ( v11 );
    v10 = v6;
  }
  v16 = v7[37];
  v17 = *(_QWORD *)(v16 + 40);
  v18 = v17 + 8LL * *(unsigned int *)(v16 + 48);
  if ( v18 != v17 )
  {
    v19 = v18 - v17 - 8;
    v20 = 0;
    a2 = (_BYTE *)(v19 >> 3);
    do
    {
      *(_DWORD *)(*(_QWORD *)(v17 + 8LL * (_QWORD)v20) + 36LL) = (_DWORD)v20;
      v21 = v20++;
    }
    while ( v21 != a2 );
  }
  v22 = v53;
  v23 = 16 * v10;
  v45 = (__int64 *)&v53[v23];
  if ( v53 != &v53[v23] )
  {
    _BitScanReverse64(&v24, v23 >> 4);
    sub_E8F9F0((__int64)v53, v45, 2LL * (int)(63 - (v24 ^ 0x3F)));
    if ( (unsigned __int64)v23 > 0x100 )
    {
      v34 = v22 + 256;
      a2 = v22 + 256;
      sub_E8F510(v22, v22 + 256);
      if ( v45 != (__int64 *)(v22 + 256) )
      {
        do
        {
          v35 = v34;
          v34 += 16;
          sub_E8F410(v35);
        }
        while ( v45 != (__int64 *)v34 );
      }
    }
    else
    {
      a2 = v45;
      sub_E8F510(v22, (char *)v45);
    }
    v25 = 16LL * (unsigned int)v54;
    v45 = (__int64 *)&v53[v25];
    if ( v53 != &v53[v25] )
    {
      v46 = (__int64)v53;
      do
      {
        v26 = *(_QWORD **)v46;
        v27 = *(_QWORD *)(v46 + 8);
        v28 = *(_QWORD *)(v44 + 168);
        v29 = **(_QWORD ***)v46;
        if ( !v29 )
        {
          if ( (*((_BYTE *)v26 + 9) & 0x70) != 0x20 || *((char *)v26 + 8) < 0 )
            BUG();
          *((_BYTE *)v26 + 8) |= 8u;
          v29 = sub_E807D0(v26[3]);
          *v26 = v29;
        }
        a2 = (_BYTE *)sub_E89FB0(v28, v29[1]);
        if ( a2 )
        {
          (*(void (__fastcall **)(_QWORD *, _BYTE *, _QWORD))(*v7 + 176LL))(v7, a2, 0);
          v36 = 0;
          v52 = 0;
          v37 = *(__int64 **)(v27 + 16);
          v50 = 0;
          v51 = 0;
          if ( v37 )
          {
            a2 = 0;
            while ( 1 )
            {
              v56[0] = v37[3];
              if ( a2 == v36 )
              {
                sub_E90980((unsigned __int64 *)&v50, a2, (const __m128i *)(v37 + 1), v56);
                v37 = (__int64 *)*v37;
                a2 = v51;
                if ( !v37 )
                  goto LABEL_42;
              }
              else
              {
                if ( a2 )
                {
                  *(__m128i *)a2 = _mm_loadu_si128((const __m128i *)(v37 + 1));
                  *((_QWORD *)a2 + 2) = v56[0];
                  a2 = v51;
                }
                a2 += 24;
                v51 = a2;
                v37 = (__int64 *)*v37;
                if ( !v37 )
                {
LABEL_42:
                  v38 = v50;
                  v39 = (__int64)a2;
                  if ( (_BYTE *)v50 != a2 )
                  {
                    _BitScanReverse64(&v40, 0xAAAAAAAAAAAAAAABLL * ((__int64)&a2[-v50] >> 3));
                    sub_E90E90(v50, (unsigned __int64)a2, 2LL * (int)(63 - (v40 ^ 0x3F)));
                    sub_E8F3B0(v38, (__int64)a2);
                    v39 = v50;
                    v47 = v51;
                    if ( v51 != (_BYTE *)v50 )
                    {
                      do
                      {
                        if ( (v26[1] & 1) != 0 )
                        {
                          v41 = (_QWORD *)*(v26 - 1);
                          v42 = *v41;
                          v43 = (int *)(v41 + 3);
                        }
                        else
                        {
                          v43 = 0;
                          v42 = 0;
                        }
                        v39 += 24;
                        v48 = v42;
                        sub_C7D030(v56);
                        sub_C7D280((int *)v56, v43, v48);
                        sub_C7D290(v56, v49);
                        v56[3] = (unsigned __int64)v26;
                        a2 = v7;
                        v56[0] = 0;
                        v56[2] = (unsigned __int64)v49[0];
                        v56[1] = 2;
                        v49[0] = (unsigned int *)v56;
                        sub_E91180(*(_QWORD **)(v39 - 8), v7, v49);
                      }
                      while ( (_BYTE *)v39 != v47 );
                      v39 = v50;
                    }
                  }
                  if ( v39 )
                  {
                    a2 = &v52[-v39];
                    j_j___libc_free_0(v39, &v52[-v39]);
                  }
                  break;
                }
              }
              v36 = v52;
            }
          }
        }
        v46 += 16;
      }
      while ( v45 != (__int64 *)v46 );
      v45 = (__int64 *)v53;
    }
  }
  result = (__int64)v45;
  if ( v45 != (__int64 *)v55 )
    return _libc_free(v45, a2);
  return result;
}
