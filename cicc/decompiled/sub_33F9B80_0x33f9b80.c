// Function: sub_33F9B80
// Address: 0x33f9b80
//
void __fastcall sub_33F9B80(__int64 a1, __int64 a2, int a3, __int64 a4, int a5, size_t a6, unsigned int a7, char a8)
{
  unsigned int v10; // ecx
  int v11; // r10d
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rax
  unsigned int v15; // edx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 *v18; // r13
  unsigned int v19; // r15d
  __int64 *v20; // rbx
  __int64 v21; // r8
  __int64 v22; // rdx
  void *v23; // r10
  __int64 v24; // rdx
  unsigned __int64 v25; // r12
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // r12
  __int64 v29; // r10
  unsigned int v30; // edx
  __int64 *v31; // rax
  __int64 *v32; // r12
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  unsigned __int64 v35; // rdi
  __int64 **v36; // rdi
  __int64 **v37; // r12
  __int64 **v38; // rbx
  __int64 *v39; // rsi
  __int64 v40; // rax
  __int64 v41; // rdx
  _BYTE *v42; // rdi
  int i; // ecx
  size_t n; // [rsp+8h] [rbp-F8h]
  void *src; // [rsp+10h] [rbp-F0h]
  unsigned int v47; // [rsp+1Ch] [rbp-E4h]
  __int64 v49; // [rsp+30h] [rbp-D0h]
  unsigned __int8 v50; // [rsp+30h] [rbp-D0h]
  char v53; // [rsp+48h] [rbp-B8h]
  __int64 *v54; // [rsp+48h] [rbp-B8h]
  _QWORD v55[2]; // [rsp+50h] [rbp-B0h] BYREF
  char v56; // [rsp+60h] [rbp-A0h]
  __int64 **v57; // [rsp+70h] [rbp-90h] BYREF
  __int64 v58; // [rsp+78h] [rbp-88h]
  _BYTE v59[16]; // [rsp+80h] [rbp-80h] BYREF
  _BYTE *v60; // [rsp+90h] [rbp-70h] BYREF
  __int64 v61; // [rsp+98h] [rbp-68h]
  _BYTE v62[96]; // [rsp+A0h] [rbp-60h] BYREF

  v47 = a6;
  if ( a4 != a2 )
  {
    v10 = a2;
    v53 = *(_BYTE *)(a2 + 32);
    v11 = v53 & 1;
    if ( (v53 & 1) != 0 )
    {
      v57 = (__int64 **)v59;
      v58 = 0x200000000LL;
      v12 = *(_QWORD *)(a1 + 720);
      v13 = *(_QWORD *)(v12 + 696);
      v14 = *(unsigned int *)(v12 + 712);
      if ( (_DWORD)v14 )
      {
        v15 = (v14 - 1) & ((v10 >> 9) ^ ((unsigned int)a2 >> 4));
        v16 = v13 + 40LL * v15;
        v17 = *(_QWORD *)v16;
        if ( a2 != *(_QWORD *)v16 )
        {
          for ( i = 1; ; i = a6 )
          {
            if ( v17 == -4096 )
              return;
            a6 = (unsigned int)(i + 1);
            v15 = (v14 - 1) & (i + v15);
            v16 = v13 + 40LL * v15;
            v17 = *(_QWORD *)v16;
            if ( a2 == *(_QWORD *)v16 )
              break;
          }
        }
        if ( v16 != v13 + 40 * v14 )
        {
          v18 = *(__int64 **)(v16 + 8);
          v54 = &v18[*(unsigned int *)(v16 + 16)];
          if ( v54 != v18 )
          {
            v19 = v11;
            while ( 1 )
            {
              v20 = (__int64 *)*v18;
              v21 = *(unsigned __int8 *)(*v18 + 62);
              if ( !(_BYTE)v21 )
                break;
LABEL_27:
              if ( v54 == ++v18 )
              {
                v36 = v57;
                v37 = &v57[(unsigned int)v58];
                if ( v37 != v57 )
                {
                  v38 = v57;
                  do
                  {
                    v39 = *v38++;
                    sub_33F99B0(a1, v39, 0, v16, v21, a6);
                  }
                  while ( v37 != v38 );
                  v36 = v57;
                }
                if ( v36 != (__int64 **)v59 )
                  _libc_free((unsigned __int64)v36);
                return;
              }
            }
            v22 = *v20;
            v23 = (void *)v20[1];
            v60 = v62;
            v61 = 0x200000000LL;
            v24 = 24 * v22;
            a6 = v24;
            v25 = 0xAAAAAAAAAAAAAAABLL * (v24 >> 3);
            if ( (unsigned __int64)v24 > 0x30 )
            {
              n = v24;
              src = v23;
              sub_C8D5F0((__int64)&v60, v62, 0xAAAAAAAAAAAAAAABLL * (v24 >> 3), 0x18u, v21, v24);
              LOBYTE(v21) = 0;
              v23 = src;
              a6 = n;
              v42 = &v60[24 * (unsigned int)v61];
            }
            else
            {
              v16 = (__int64)v62;
              if ( !v24 )
              {
LABEL_12:
                LODWORD(v61) = v24 + v25;
                v26 = v16 + 24LL * (unsigned int)(v24 + v25);
                if ( v26 != v16 )
                {
                  v27 = v16;
                  do
                  {
                    if ( !*(_DWORD *)v27 && a2 == *(_QWORD *)(v27 + 8) && a3 == *(_DWORD *)(v27 + 16) )
                    {
                      *(_QWORD *)(v27 + 8) = a4;
                      v21 = v19;
                      *(_DWORD *)(v27 + 16) = a5;
                    }
                    v27 += 24;
                  }
                  while ( v26 != v27 );
                  if ( (_BYTE)v21 )
                  {
                    v28 = v20[4];
                    v29 = v20[5];
                    if ( !a7 )
                      goto LABEL_19;
                    v49 = v20[5];
                    sub_AF47B0((__int64)v55, *(unsigned __int64 **)(v29 + 16), *(unsigned __int64 **)(v29 + 24));
                    if ( !v56 || (unsigned __int64)(a7 + v47) <= v55[0] )
                    {
                      v40 = sub_B0E470(v49, v47, a7);
                      v55[1] = v41;
                      v55[0] = v40;
                      if ( (_BYTE)v41 )
                      {
                        v29 = v55[0];
                        v16 = (__int64)v60;
LABEL_19:
                        v30 = *((_DWORD *)v20 + 14);
                        if ( *(_DWORD *)(a4 + 72) >= v30 )
                          v30 = *(_DWORD *)(a4 + 72);
                        v31 = sub_33E4BC0(
                                a1,
                                v28,
                                v29,
                                (const void *)v16,
                                (unsigned int)v61,
                                *((_BYTE *)v20 + 60),
                                (const void *)v20[3],
                                v20[2],
                                v20 + 6,
                                v30,
                                *((_BYTE *)v20 + 61));
                        v16 = HIDWORD(v58);
                        v32 = v31;
                        v33 = (unsigned int)v58;
                        v34 = (unsigned int)v58 + 1LL;
                        if ( v34 > HIDWORD(v58) )
                        {
                          sub_C8D5F0((__int64)&v57, v59, v34, 8u, v21, a6);
                          v33 = (unsigned int)v58;
                        }
                        v57[v33] = v32;
                        LODWORD(v58) = v58 + 1;
                        if ( a8 )
                          *((_WORD *)v20 + 31) = 257;
                        v35 = (unsigned __int64)v60;
                        if ( v60 == v62 )
                          goto LABEL_27;
LABEL_26:
                        _libc_free(v35);
                        goto LABEL_27;
                      }
                    }
                    v16 = (__int64)v60;
                  }
                }
                if ( (_BYTE *)v16 == v62 )
                  goto LABEL_27;
                v35 = v16;
                goto LABEL_26;
              }
              v42 = v62;
            }
            v50 = v21;
            memcpy(v42, v23, a6);
            v16 = (__int64)v60;
            LODWORD(v24) = v61;
            v21 = v50;
            goto LABEL_12;
          }
        }
      }
    }
  }
}
