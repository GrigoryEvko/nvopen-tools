// Function: sub_32469C0
// Address: 0x32469c0
//
void __fastcall sub_32469C0(__int64 **a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned __int64 v9; // rdx
  unsigned int v10; // edx
  __int64 v11; // rax
  __int64 *v12; // r14
  __int64 v13; // rax
  __int64 v14; // r13
  unsigned int v15; // ecx
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 *v18; // rax
  __int64 v19; // rdx
  _BYTE *v20; // rdi
  char *v21; // r15
  __int64 v22; // r13
  char *v23; // r14
  unsigned __int64 v24; // rax
  __int64 v25; // rax
  char *v26; // rbx
  __int64 v27; // r14
  __int64 *v28; // rsi
  __int64 v29; // rdi
  void (*v30)(); // r8
  __int64 v31; // rsi
  __int64 v32; // rdx
  void (*v33)(); // r8
  __int64 v34; // rdx
  unsigned __int64 v35; // r13
  __int64 v36; // rax
  char *v37; // rax
  char *i; // rdx
  unsigned int v39; // ecx
  __int64 v40; // rsi
  unsigned int v41; // eax
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  unsigned int v46; // r13d
  char *v47; // r12
  char *v48; // r14
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 *v51; // rdx
  __int64 v52; // rdi
  __int64 v53; // rax
  __int64 *v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rdx
  char *v57; // r13
  __int64 v58; // rcx
  __int64 v59; // rdx
  char *v60; // rax
  char *v61; // rsi
  char *v62; // rsi
  char *v65; // [rsp+28h] [rbp-278h]
  _QWORD v66[4]; // [rsp+30h] [rbp-270h] BYREF
  __int16 v67; // [rsp+50h] [rbp-250h]
  void *src; // [rsp+60h] [rbp-240h] BYREF
  __int64 v69; // [rsp+68h] [rbp-238h]
  _BYTE v70[560]; // [rsp+70h] [rbp-230h] BYREF

  if ( *((_DWORD *)a1 + 3) )
  {
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a2 + 224) + 176LL))(*(_QWORD *)(a2 + 224), a3, 0);
    v9 = *((unsigned int *)a1 + 3);
    src = v70;
    v69 = 0x4000000000LL;
    if ( v9 > 0x40 )
    {
      sub_C8D5F0((__int64)&src, v70, v9, 8u, v7, v8);
      v10 = *((_DWORD *)a1 + 2);
      if ( v10 )
        goto LABEL_4;
    }
    else
    {
      v10 = *((_DWORD *)a1 + 2);
      if ( v10 )
      {
LABEL_4:
        v11 = **a1;
        v12 = *a1;
        if ( v11 == -8 || !v11 )
        {
          do
          {
            do
            {
              v13 = v12[1];
              ++v12;
            }
            while ( !v13 );
          }
          while ( v13 == -8 );
        }
        v14 = (__int64)&(*a1)[v10];
        v15 = v69;
        if ( v12 != (__int64 *)v14 )
        {
          while ( 1 )
          {
            v16 = v15;
            v17 = *v12;
            if ( (unsigned __int64)v15 + 1 > HIDWORD(v69) )
            {
              sub_C8D5F0((__int64)&src, v70, v15 + 1LL, 8u, v7, v8);
              v16 = (unsigned int)v69;
            }
            *((_QWORD *)src + v16) = v17;
            v15 = v69 + 1;
            v18 = v12 + 1;
            LODWORD(v69) = v69 + 1;
            v19 = v12[1];
            if ( v19 == -8 )
              break;
LABEL_13:
            if ( !v19 )
              goto LABEL_12;
            if ( v18 == (__int64 *)v14 )
              goto LABEL_22;
            v12 = v18;
          }
          do
          {
LABEL_12:
            v19 = v18[1];
            ++v18;
          }
          while ( v19 == -8 );
          goto LABEL_13;
        }
LABEL_22:
        v21 = (char *)src;
        v22 = 8LL * v15;
        v23 = (char *)src + v22;
        if ( src != (char *)src + v22 )
        {
          _BitScanReverse64(&v24, v22 >> 3);
          sub_32466D0((char *)src, (__int64 *)((char *)src + v22), 2LL * (int)(63 - (v24 ^ 0x3F)));
          if ( (unsigned __int64)v22 > 0x80 )
          {
            v57 = v21 + 128;
            sub_3246610(v21, v21 + 128);
            if ( v23 != v21 + 128 )
            {
              do
              {
                while ( 1 )
                {
                  v58 = *(_QWORD *)v57;
                  v59 = *((_QWORD *)v57 - 1);
                  v60 = v57 - 8;
                  if ( *(_QWORD *)(v59 + 16) > *(_QWORD *)(*(_QWORD *)v57 + 16LL) )
                    break;
                  v62 = v57;
                  v57 += 8;
                  *(_QWORD *)v62 = v58;
                  if ( v23 == v57 )
                    goto LABEL_25;
                }
                do
                {
                  *((_QWORD *)v60 + 1) = v59;
                  v61 = v60;
                  v59 = *((_QWORD *)v60 - 1);
                  v60 -= 8;
                }
                while ( *(_QWORD *)(v58 + 16) < *(_QWORD *)(v59 + 16) );
                v57 += 8;
                *(_QWORD *)v61 = v58;
              }
              while ( v23 != v57 );
            }
          }
          else
          {
            sub_3246610(v21, v23);
          }
LABEL_25:
          v65 = (char *)src + 8 * (unsigned int)v69;
          if ( src != v65 )
          {
            v25 = a2;
            v26 = (char *)src;
            v27 = v25;
            do
            {
              v29 = *(_QWORD *)(v27 + 224);
              if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(v27 + 200) + 544LL) - 42) <= 1 )
              {
                v30 = *(void (**)())(*(_QWORD *)v29 + 120LL);
                v31 = **(_QWORD **)v26;
                v32 = *(_QWORD *)v26 + 32LL;
                v67 = 261;
                v66[0] = v32;
                v66[1] = v31;
                if ( v30 != nullsub_98 )
                {
                  ((void (__fastcall *)(__int64, _QWORD *, __int64))v30)(v29, v66, 1);
                  v29 = *(_QWORD *)(v27 + 224);
                }
              }
              if ( *((_BYTE *)a1 + 60) )
              {
                (*(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v29 + 208LL))(
                  v29,
                  *(_QWORD *)(*(_QWORD *)v26 + 8LL),
                  0);
                v29 = *(_QWORD *)(v27 + 224);
              }
              v33 = *(void (**)())(*(_QWORD *)v29 + 120LL);
              v34 = *(_QWORD *)v26 + 16LL;
              v66[0] = "string offset=";
              v66[2] = v34;
              v67 = 2819;
              if ( v33 != nullsub_98 )
              {
                ((void (__fastcall *)(__int64, _QWORD *, __int64))v33)(v29, v66, 1);
                v29 = *(_QWORD *)(v27 + 224);
              }
              v28 = *(__int64 **)v26;
              v26 += 8;
              (*(void (__fastcall **)(__int64, __int64 *, __int64))(*(_QWORD *)v29 + 512LL))(v29, v28 + 4, *v28 + 1);
            }
            while ( v65 != v26 );
            a2 = v27;
          }
        }
        if ( a4 )
        {
          v35 = *((unsigned int *)a1 + 14);
          v36 = (unsigned int)v69;
          if ( v35 != (unsigned int)v69 )
          {
            if ( v35 >= (unsigned int)v69 )
            {
              if ( v35 > HIDWORD(v69) )
              {
                sub_C8D5F0((__int64)&src, v70, *((unsigned int *)a1 + 14), 8u, v7, v8);
                v36 = (unsigned int)v69;
              }
              v37 = (char *)src + 8 * v36;
              for ( i = (char *)src + 8 * v35; i != v37; v37 += 8 )
              {
                if ( v37 )
                  *(_QWORD *)v37 = 0;
              }
            }
            LODWORD(v69) = v35;
          }
          v39 = *((_DWORD *)a1 + 2);
          if ( v39 )
          {
            v50 = **a1;
            v51 = *a1;
            if ( v50 != -8 )
              goto LABEL_55;
            do
            {
              do
              {
                v50 = v51[1];
                ++v51;
              }
              while ( v50 == -8 );
LABEL_55:
              ;
            }
            while ( !v50 );
            v52 = (__int64)&(*a1)[v39];
            if ( v51 != (__int64 *)v52 )
            {
              while ( 1 )
              {
                v53 = *(unsigned int *)(*v51 + 24);
                if ( (_DWORD)v53 != -1 )
                  *((_QWORD *)src + v53) = *v51;
                v54 = v51 + 1;
                v55 = v51[1];
                if ( !v55 || v55 == -8 )
                {
                  do
                  {
                    do
                    {
                      v56 = v54[1];
                      ++v54;
                    }
                    while ( v56 == -8 );
                  }
                  while ( !v56 );
                }
                if ( v54 == (__int64 *)v52 )
                  break;
                v51 = v54;
              }
            }
          }
          v40 = a4;
          (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a2 + 224) + 176LL))(
            *(_QWORD *)(a2 + 224),
            a4,
            0);
          v41 = sub_31DF6B0(a2);
          v20 = src;
          v46 = v41;
          v47 = (char *)src + 8 * (unsigned int)v69;
          if ( v47 == src )
            goto LABEL_17;
          v48 = (char *)src;
          do
          {
            v49 = *(_QWORD *)v48;
            if ( a5 )
            {
              sub_31F0E70(a2, v40, v42, v43, v44, v45, *(_QWORD *)(v49 + 8), *(_QWORD *)(v49 + 16));
            }
            else
            {
              v40 = *(_QWORD *)(v49 + 16);
              (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a2 + 224) + 536LL))(
                *(_QWORD *)(a2 + 224),
                v40,
                v46);
            }
            v48 += 8;
          }
          while ( v47 != v48 );
        }
        v20 = src;
LABEL_17:
        if ( v20 != v70 )
          _libc_free((unsigned __int64)v20);
        return;
      }
    }
    v15 = v69;
    goto LABEL_22;
  }
}
