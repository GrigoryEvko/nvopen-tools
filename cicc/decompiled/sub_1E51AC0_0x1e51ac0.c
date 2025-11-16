// Function: sub_1E51AC0
// Address: 0x1e51ac0
//
__int64 __fastcall sub_1E51AC0(__int64 a1, signed int a2, unsigned int a3, __int64 a4, char a5)
{
  __int64 v7; // rbx
  __int64 v8; // rdi
  __int64 v10; // rax
  unsigned int *v11; // r12
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // rbx
  char v16; // r13
  unsigned int *v17; // r15
  char v18; // al
  unsigned int v19; // eax
  __int64 v20; // rsi
  char v21; // r10
  int v22; // r13d
  unsigned int v23; // r8d
  int v24; // edx
  __int64 v25; // rax
  __int64 v26; // rsi
  int v27; // ecx
  __int64 v28; // r10
  unsigned int v29; // edx
  __int64 *v30; // rdi
  __int64 v31; // r9
  int **v33; // rax
  int *v34; // r12
  __int64 v35; // r14
  __int64 v36; // rax
  __int64 v37; // r15
  __int64 v38; // r15
  __int64 *v39; // rdx
  __int64 *v40; // rsi
  __int64 v41; // r11
  __int64 v42; // rax
  __int64 v43; // r12
  int v44; // r9d
  __int64 v45; // rax
  __int64 i; // rcx
  int v47; // r13d
  unsigned __int64 v48; // rdi
  unsigned int v49; // edx
  __int64 v50; // rsi
  unsigned int v51; // edx
  __int64 v52; // rax
  int v53; // edi
  int v54; // r11d
  __int64 v55; // rdi
  __int64 v56; // rsi
  __int64 v57; // rsi
  __int64 v59; // [rsp+8h] [rbp-B8h]
  unsigned int v61; // [rsp+14h] [rbp-ACh]
  __int64 v62; // [rsp+28h] [rbp-98h] BYREF
  __int64 v63; // [rsp+30h] [rbp-90h] BYREF
  __int64 v64; // [rsp+38h] [rbp-88h]
  __int64 v65; // [rsp+40h] [rbp-80h]
  __int64 v66; // [rsp+48h] [rbp-78h]
  __int64 v67; // [rsp+50h] [rbp-70h]
  __int64 v68; // [rsp+58h] [rbp-68h]
  __int64 v69; // [rsp+60h] [rbp-60h]
  char v70; // [rsp+68h] [rbp-58h]
  __int64 v71; // [rsp+6Ch] [rbp-54h]
  __int64 v72; // [rsp+74h] [rbp-4Ch]
  __int64 v73; // [rsp+80h] [rbp-40h]
  int v74; // [rsp+88h] [rbp-38h]

  v7 = a1;
  v8 = a1 + 8;
  v62 = **(_QWORD **)(v8 - 8) + 272LL * a2;
  sub_1E51470(v8, &v62);
  *(_QWORD *)(*(_QWORD *)(v7 + 64) + 8LL * ((unsigned int)a2 >> 6)) |= 1LL << a2;
  v10 = *(_QWORD *)(v7 + 824) + 32LL * a2;
  v59 = 32LL * a2;
  v11 = *(unsigned int **)v10;
  v12 = *(unsigned int *)(v10 + 8);
  if ( v11 != &v11[v12] )
  {
    v13 = v7;
    v14 = a3;
    v15 = a4;
    v16 = 0;
    v17 = &v11[v12];
    do
    {
      v19 = *(_DWORD *)(v13 + 1352);
      v20 = *v11;
      if ( v19 > 5 )
        break;
      if ( (int)v20 >= (int)v14 )
      {
        if ( (_DWORD)v20 == (_DWORD)v14 )
        {
          v38 = v15;
          v7 = v13;
          if ( !a5 )
          {
            v39 = *(__int64 **)(v13 + 48);
            v40 = *(__int64 **)(v13 + 40);
            v63 = 0;
            v64 = 0;
            v65 = 0;
            v66 = 0;
            v67 = 0;
            v68 = 0;
            v69 = 0;
            sub_1E51930((__int64)&v63, v40, v39);
            v41 = v67;
            v70 = 1;
            v71 = 0;
            v72 = 0;
            v42 = (v68 - v67) >> 3;
            v73 = 0;
            v74 = 0;
            if ( (_DWORD)v42 )
            {
              v43 = v67 + 8LL * (unsigned int)(v42 - 1) + 8;
              v44 = v66 - 1;
              do
              {
                v45 = *(_QWORD *)(*(_QWORD *)v41 + 112LL);
                for ( i = v45 + 16LL * *(unsigned int *)(*(_QWORD *)v41 + 120LL); i != v45; v45 += 16 )
                {
                  if ( (_DWORD)v66 )
                  {
                    v47 = 1;
                    v48 = *(_QWORD *)v45 & 0xFFFFFFFFFFFFFFF8LL;
                    v49 = v44 & (((unsigned int)*(_QWORD *)v45 >> 4) ^ ((unsigned int)*(_QWORD *)v45 >> 9));
                    v50 = *(_QWORD *)(v64 + 8LL * v49);
                    if ( v48 == v50 )
                    {
LABEL_30:
                      v74 += *(_DWORD *)(v45 + 12);
                    }
                    else
                    {
                      while ( v50 != -8 )
                      {
                        v49 = v44 & (v47 + v49);
                        v50 = *(_QWORD *)(v64 + 8LL * v49);
                        if ( v48 == v50 )
                          goto LABEL_30;
                        ++v47;
                      }
                    }
                  }
                }
                v41 += 8;
              }
              while ( v43 != v41 );
            }
            v51 = *(_DWORD *)(v38 + 8);
            if ( v51 >= *(_DWORD *)(v38 + 12) )
            {
              sub_1E44B20(v38, 0);
              v51 = *(_DWORD *)(v38 + 8);
            }
            v52 = *(_QWORD *)v38 + 96LL * v51;
            if ( v52 )
            {
              *(_QWORD *)(v52 + 16) = 0;
              *(_QWORD *)(v52 + 8) = 0;
              *(_DWORD *)(v52 + 24) = 0;
              *(_QWORD *)v52 = 1;
              ++v63;
              *(_QWORD *)(v52 + 8) = v64;
              *(_QWORD *)(v52 + 16) = v65;
              *(_DWORD *)(v52 + 24) = v66;
              v64 = 0;
              v65 = 0;
              LODWORD(v66) = 0;
              *(_QWORD *)(v52 + 32) = v67;
              *(_QWORD *)(v52 + 40) = v68;
              *(_QWORD *)(v52 + 48) = v69;
              *(_BYTE *)(v52 + 56) = v70;
              *(_QWORD *)(v52 + 60) = v71;
              *(_QWORD *)(v52 + 68) = v72;
              *(_QWORD *)(v52 + 80) = v73;
              *(_DWORD *)(v52 + 88) = v74;
              ++*(_DWORD *)(v38 + 8);
            }
            else
            {
              v55 = v67;
              v56 = v69;
              *(_DWORD *)(v38 + 8) = v51 + 1;
              v57 = v56 - v55;
              if ( v55 )
                j_j___libc_free_0(v55, v57);
            }
            j___libc_free_0(v64);
            v19 = *(_DWORD *)(v7 + 1352);
          }
          *(_DWORD *)(v7 + 1352) = v19 + 1;
          goto LABEL_11;
        }
        if ( (*(_QWORD *)(*(_QWORD *)(v13 + 64) + 8LL * ((unsigned int)v20 >> 6)) & (1LL << v20)) == 0 )
        {
          v61 = v14;
          v18 = sub_1E51AC0(v13, v20, v14, v15, (unsigned __int8)(a5 | (a2 > (int)v20)));
          v14 = v61;
          if ( v18 )
            v16 = v18;
        }
      }
      ++v11;
    }
    while ( v17 != v11 );
    v21 = v16;
    v7 = v13;
    v22 = v14;
    if ( v21 )
    {
LABEL_11:
      sub_1E42470(v7, a2);
      v23 = 1;
      goto LABEL_12;
    }
    v33 = (int **)(*(_QWORD *)(v13 + 824) + v59);
    v34 = *v33;
    v35 = (__int64)&(*v33)[*((unsigned int *)v33 + 2)];
    if ( (int *)v35 != *v33 )
    {
      do
      {
        while ( 1 )
        {
          v36 = *v34;
          if ( v22 <= (int)v36 )
          {
            v37 = 72 * v36;
            if ( !sub_1E46A20(72 * v36 + *(_QWORD *)(v13 + 88), v62) )
              break;
          }
          if ( (int *)v35 == ++v34 )
            goto LABEL_22;
        }
        ++v34;
        sub_1E46B00((__int64)&v63, v37 + *(_QWORD *)(v13 + 88), v62);
      }
      while ( (int *)v35 != v34 );
    }
  }
LABEL_22:
  v23 = 0;
LABEL_12:
  v24 = *(_DWORD *)(v7 + 32);
  v25 = *(_QWORD *)(v7 + 48);
  if ( v24 )
  {
    v26 = *(_QWORD *)(v25 - 8);
    v27 = v24 - 1;
    v28 = *(_QWORD *)(v7 + 16);
    v29 = (v24 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
    v30 = (__int64 *)(v28 + 8LL * v29);
    v31 = *v30;
    if ( *v30 == v26 )
    {
LABEL_14:
      *v30 = -16;
      v25 = *(_QWORD *)(v7 + 48);
      --*(_DWORD *)(v7 + 24);
      ++*(_DWORD *)(v7 + 28);
    }
    else
    {
      v53 = 1;
      while ( v31 != -8 )
      {
        v54 = v53 + 1;
        v29 = v27 & (v53 + v29);
        v30 = (__int64 *)(v28 + 8LL * v29);
        v31 = *v30;
        if ( v26 == *v30 )
          goto LABEL_14;
        v53 = v54;
      }
    }
  }
  *(_QWORD *)(v7 + 48) = v25 - 8;
  return v23;
}
