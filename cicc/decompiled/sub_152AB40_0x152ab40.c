// Function: sub_152AB40
// Address: 0x152ab40
//
void __fastcall sub_152AB40(__int64 *a1, _QWORD *a2, unsigned __int64 a3, __int64 a4)
{
  _QWORD *v6; // rbx
  unsigned int v7; // edx
  unsigned int v8; // r15d
  int v9; // r12d
  __int64 v10; // rbx
  __int64 v11; // rdx
  unsigned int v12; // edx
  _QWORD *v13; // rbx
  _QWORD *v14; // r14
  __int64 v15; // rbx
  __int64 v16; // r12
  volatile signed __int32 *v17; // r13
  signed __int32 v18; // eax
  signed __int32 v19; // eax
  _QWORD *v20; // rbx
  _QWORD *v21; // r14
  __int64 v22; // rbx
  __int64 v23; // r12
  volatile signed __int32 *v24; // r13
  signed __int32 v25; // eax
  signed __int32 v26; // eax
  __int64 v27; // rbx
  __int64 v28; // r8
  __int64 v29; // r12
  volatile signed __int32 *v30; // rdi
  _QWORD *v31; // rbx
  __int64 v32; // rdi
  unsigned int v33; // eax
  __int64 v34; // rax
  unsigned __int64 v35; // rdx
  const void *v36; // r15
  size_t v37; // r12
  unsigned __int8 *v38; // r15
  __int64 v39; // rbx
  __int64 v40; // r12
  __int64 v41; // rax
  __int64 v42; // r8
  __int64 v43; // rdx
  _QWORD *v44; // rdx
  __int64 v45; // rdx
  _QWORD *v46; // rdx
  __int64 v47; // rdx
  _QWORD *v48; // rdx
  __int64 v49; // rdx
  _QWORD *v50; // rdx
  _QWORD *v51; // rdi
  unsigned int v52; // eax
  unsigned int v53; // esi
  __int64 v54; // rbx
  int v55; // r8d
  __int64 v56; // rax
  _QWORD *v59; // [rsp+30h] [rbp-1B0h]
  __int64 v60; // [rsp+30h] [rbp-1B0h]
  __int64 v61; // [rsp+30h] [rbp-1B0h]
  __int64 v62; // [rsp+30h] [rbp-1B0h]
  _QWORD *v63; // [rsp+38h] [rbp-1A8h]
  _QWORD *v64; // [rsp+38h] [rbp-1A8h]
  _QWORD *v65; // [rsp+38h] [rbp-1A8h]
  unsigned int v66; // [rsp+38h] [rbp-1A8h]
  int v67; // [rsp+38h] [rbp-1A8h]
  __int64 v68; // [rsp+38h] [rbp-1A8h]
  __int64 v69; // [rsp+38h] [rbp-1A8h]
  __int64 v70; // [rsp+38h] [rbp-1A8h]
  unsigned __int64 v71; // [rsp+40h] [rbp-1A0h] BYREF
  volatile signed __int32 *v72; // [rsp+48h] [rbp-198h]
  int v73; // [rsp+50h] [rbp-190h]
  __int64 v74; // [rsp+58h] [rbp-188h]
  __int64 v75; // [rsp+60h] [rbp-180h]
  __int64 v76; // [rsp+68h] [rbp-178h]
  _QWORD *v77; // [rsp+70h] [rbp-170h]
  _QWORD *v78; // [rsp+78h] [rbp-168h]
  __int64 v79; // [rsp+80h] [rbp-160h]
  _QWORD *v80; // [rsp+88h] [rbp-158h]
  _QWORD *v81; // [rsp+90h] [rbp-150h]
  __int64 v82; // [rsp+98h] [rbp-148h]
  unsigned __int64 v83; // [rsp+A0h] [rbp-140h] BYREF
  __int64 v84; // [rsp+A8h] [rbp-138h]
  _BYTE v85[304]; // [rsp+B0h] [rbp-130h] BYREF

  if ( a3 )
  {
    v83 = 35;
    sub_1525CA0(a4, &v83);
    v83 = a3;
    sub_1525CA0(a4, &v83);
    v71 = (unsigned __int64)&v83;
    v83 = (unsigned __int64)v85;
    v84 = 0x10000000000LL;
    v72 = 0;
    v73 = 2;
    v74 = 0;
    v75 = 0;
    v76 = 0;
    v77 = 0;
    v78 = 0;
    v79 = 0;
    v80 = 0;
    v81 = 0;
    v82 = 0;
    v59 = &a2[a3];
    if ( v59 != a2 )
    {
      v6 = a2;
      do
      {
        sub_161E970(*v6);
        v8 = v7;
        if ( v7 > 0x1F )
        {
          v63 = v6;
          do
          {
            while ( 1 )
            {
              v9 = HIDWORD(v72) | ((v8 & 0x1F | 0x20) << (char)v72);
              HIDWORD(v72) = v9;
              if ( (unsigned int)((_DWORD)v72 + 6) > 0x1F )
                break;
              v8 >>= 5;
              LODWORD(v72) = (_DWORD)v72 + 6;
              if ( v8 <= 0x1F )
                goto LABEL_14;
            }
            v10 = v71;
            v11 = *(unsigned int *)(v71 + 8);
            if ( (unsigned __int64)*(unsigned int *)(v71 + 12) - v11 <= 3 )
            {
              sub_16CD150(v71, v71 + 16, v11 + 4, 1);
              v11 = *(unsigned int *)(v10 + 8);
            }
            *(_DWORD *)(*(_QWORD *)v10 + v11) = v9;
            v12 = 0;
            *(_DWORD *)(v10 + 8) += 4;
            if ( (_DWORD)v72 )
              v12 = (v8 & 0x1F | 0x20) >> (32 - (_BYTE)v72);
            v8 >>= 5;
            HIDWORD(v72) = v12;
            LODWORD(v72) = ((_BYTE)v72 + 6) & 0x1F;
          }
          while ( v8 > 0x1F );
LABEL_14:
          v6 = v63;
        }
        ++v6;
        sub_1524D80(&v71, v8, 6);
      }
      while ( v59 != v6 );
      if ( (_DWORD)v72 )
      {
        v54 = v71;
        v55 = HIDWORD(v72);
        v56 = *(unsigned int *)(v71 + 8);
        if ( (unsigned __int64)*(unsigned int *)(v71 + 12) - v56 <= 3 )
        {
          v67 = HIDWORD(v72);
          sub_16CD150(v71, v71 + 16, v56 + 4, 1);
          v56 = *(unsigned int *)(v54 + 8);
          v55 = v67;
        }
        *(_DWORD *)(*(_QWORD *)v54 + v56) = v55;
        *(_DWORD *)(v54 + 8) += 4;
        v72 = 0;
      }
      v13 = v80;
      v64 = v81;
      if ( v81 != v80 )
      {
        v14 = v80;
        do
        {
          v15 = v14[2];
          v16 = v14[1];
          if ( v15 != v16 )
          {
            do
            {
              while ( 1 )
              {
                v17 = *(volatile signed __int32 **)(v16 + 8);
                if ( v17 )
                {
                  if ( &_pthread_key_create )
                  {
                    v18 = _InterlockedExchangeAdd(v17 + 2, 0xFFFFFFFF);
                  }
                  else
                  {
                    v18 = *((_DWORD *)v17 + 2);
                    *((_DWORD *)v17 + 2) = v18 - 1;
                  }
                  if ( v18 == 1 )
                  {
                    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v17 + 16LL))(v17);
                    if ( &_pthread_key_create )
                    {
                      v19 = _InterlockedExchangeAdd(v17 + 3, 0xFFFFFFFF);
                    }
                    else
                    {
                      v19 = *((_DWORD *)v17 + 3);
                      *((_DWORD *)v17 + 3) = v19 - 1;
                    }
                    if ( v19 == 1 )
                      break;
                  }
                }
                v16 += 16;
                if ( v15 == v16 )
                  goto LABEL_30;
              }
              v16 += 16;
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v17 + 24LL))(v17);
            }
            while ( v15 != v16 );
LABEL_30:
            v16 = v14[1];
          }
          if ( v16 )
            j_j___libc_free_0(v16, v14[3] - v16);
          v14 += 4;
        }
        while ( v14 != v64 );
        v13 = v80;
      }
      if ( v13 )
        j_j___libc_free_0(v13, v82 - (_QWORD)v13);
    }
    v20 = v77;
    v65 = v78;
    if ( v78 != v77 )
    {
      v21 = v77;
      do
      {
        v22 = v21[3];
        v23 = v21[2];
        if ( v22 != v23 )
        {
          do
          {
            while ( 1 )
            {
              v24 = *(volatile signed __int32 **)(v23 + 8);
              if ( v24 )
              {
                if ( &_pthread_key_create )
                {
                  v25 = _InterlockedExchangeAdd(v24 + 2, 0xFFFFFFFF);
                }
                else
                {
                  v25 = *((_DWORD *)v24 + 2);
                  *((_DWORD *)v24 + 2) = v25 - 1;
                }
                if ( v25 == 1 )
                {
                  (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v24 + 16LL))(v24);
                  if ( &_pthread_key_create )
                  {
                    v26 = _InterlockedExchangeAdd(v24 + 3, 0xFFFFFFFF);
                  }
                  else
                  {
                    v26 = *((_DWORD *)v24 + 3);
                    *((_DWORD *)v24 + 3) = v26 - 1;
                  }
                  if ( v26 == 1 )
                    break;
                }
              }
              v23 += 16;
              if ( v22 == v23 )
                goto LABEL_50;
            }
            v23 += 16;
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v24 + 24LL))(v24);
          }
          while ( v22 != v23 );
LABEL_50:
          v23 = v21[2];
        }
        if ( v23 )
          j_j___libc_free_0(v23, v21[4] - v23);
        v21 += 5;
      }
      while ( v65 != v21 );
      v20 = v77;
    }
    if ( v20 )
      j_j___libc_free_0(v20, v79 - (_QWORD)v20);
    v27 = v75;
    v28 = v74;
    if ( v75 != v74 )
    {
      v29 = v74;
      do
      {
        v30 = *(volatile signed __int32 **)(v29 + 8);
        if ( v30 )
          sub_A191D0(v30);
        v29 += 16;
      }
      while ( v27 != v29 );
      v28 = v74;
    }
    if ( v28 )
      j_j___libc_free_0(v28, v76 - v28);
    v71 = (unsigned int)v84;
    sub_1525CA0(a4, &v71);
    v31 = a2;
    if ( v59 == a2 )
    {
      v33 = v84;
    }
    else
    {
      do
      {
        v34 = sub_161E970(*v31);
        v32 = (unsigned int)v84;
        v36 = (const void *)v34;
        v37 = v35;
        if ( v35 > HIDWORD(v84) - (unsigned __int64)(unsigned int)v84 )
        {
          sub_16CD150(&v83, v85, (unsigned int)v84 + v35, 1);
          v32 = (unsigned int)v84;
        }
        if ( v37 )
        {
          memcpy((void *)(v83 + v32), v36, v37);
          LODWORD(v32) = v84;
        }
        ++v31;
        LODWORD(v84) = v32 + v37;
        v33 = v32 + v37;
      }
      while ( v59 != v31 );
    }
    v38 = (unsigned __int8 *)v83;
    v39 = v33;
    v40 = *a1;
    v41 = sub_22077B0(544);
    if ( v41 )
    {
      v42 = v41 + 16;
      *(_QWORD *)(v41 + 8) = 0x100000001LL;
      *(_QWORD *)(v41 + 24) = 0x2000000000LL;
      *(_QWORD *)v41 = &unk_49ECD20;
      *(_QWORD *)(v41 + 16) = v41 + 32;
      v43 = 0;
    }
    else
    {
      v43 = MEMORY[0x18];
      v42 = 16;
      if ( MEMORY[0x18] >= MEMORY[0x1C] )
      {
        sub_16CD150(16, 32, 0, 16);
        v43 = MEMORY[0x18];
        v42 = 16;
        v41 = 0;
      }
    }
    v44 = (_QWORD *)(*(_QWORD *)(v41 + 16) + 16 * v43);
    *v44 = 35;
    v44[1] = 1;
    v45 = (unsigned int)(*(_DWORD *)(v41 + 24) + 1);
    *(_DWORD *)(v41 + 24) = v45;
    if ( *(_DWORD *)(v41 + 28) <= (unsigned int)v45 )
    {
      v61 = v41;
      v69 = v42;
      sub_16CD150(v42, v41 + 32, 0, 16);
      v41 = v61;
      v42 = v69;
      v45 = *(unsigned int *)(v61 + 24);
    }
    v46 = (_QWORD *)(*(_QWORD *)(v41 + 16) + 16 * v45);
    *v46 = 6;
    v46[1] = 4;
    v47 = (unsigned int)(*(_DWORD *)(v41 + 24) + 1);
    *(_DWORD *)(v41 + 24) = v47;
    if ( *(_DWORD *)(v41 + 28) <= (unsigned int)v47 )
    {
      v62 = v41;
      v70 = v42;
      sub_16CD150(v42, v41 + 32, 0, 16);
      v41 = v62;
      v42 = v70;
      v47 = *(unsigned int *)(v62 + 24);
    }
    v48 = (_QWORD *)(*(_QWORD *)(v41 + 16) + 16 * v47);
    *v48 = 6;
    v48[1] = 4;
    v49 = (unsigned int)(*(_DWORD *)(v41 + 24) + 1);
    *(_DWORD *)(v41 + 24) = v49;
    if ( *(_DWORD *)(v41 + 28) <= (unsigned int)v49 )
    {
      v60 = v41;
      v68 = v42;
      sub_16CD150(v42, v41 + 32, 0, 16);
      v41 = v60;
      v42 = v68;
      v49 = *(unsigned int *)(v60 + 24);
    }
    v50 = (_QWORD *)(*(_QWORD *)(v41 + 16) + 16 * v49);
    *v50 = 0;
    v50[1] = 10;
    v51 = (_QWORD *)*a1;
    ++*(_DWORD *)(v41 + 24);
    v71 = v42;
    v72 = (volatile signed __int32 *)v41;
    v52 = sub_15271D0(v51, (__int64 *)&v71);
    v53 = v52;
    if ( v72 )
    {
      v66 = v52;
      sub_A191D0(v72);
      v53 = v66;
    }
    BYTE4(v71) = 0;
    sub_152A250(v40, v53, *(_QWORD *)a4, *(unsigned int *)(a4 + 8), v38, v39, (__int64)&v71);
    *(_DWORD *)(a4 + 8) = 0;
    if ( (_BYTE *)v83 != v85 )
      _libc_free(v83);
  }
}
