// Function: sub_1C59C20
// Address: 0x1c59c20
//
__int64 __fastcall sub_1C59C20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  _QWORD *v5; // rax
  __int64 v6; // rax
  int v7; // r8d
  unsigned int v8; // eax
  __int64 *v9; // rcx
  __int64 v10; // rdi
  __int64 v11; // rax
  int v12; // r9d
  unsigned int v13; // r15d
  __int64 v14; // r8
  __int64 *v15; // rax
  unsigned int v16; // esi
  __int64 v17; // rdx
  unsigned int v18; // eax
  __int64 *v19; // r8
  __int64 v20; // rdi
  unsigned __int64 v21; // rax
  __int64 v22; // rbx
  int v23; // r13d
  __int64 *v24; // rax
  __int64 v25; // r12
  _QWORD *v26; // r14
  __int64 v27; // rdx
  __int64 *v28; // rdi
  unsigned int v29; // eax
  __int64 v30; // rcx
  __int64 v31; // rsi
  _QWORD *v32; // rax
  __int64 v34; // rdx
  int v35; // r8d
  __int64 *v36; // rdi
  unsigned int v37; // r9d
  __int64 *v38; // rsi
  __int64 *v39; // r11
  int v40; // edi
  __int64 v41; // rdx
  int v42; // ecx
  int v43; // ebx
  __int64 *v44; // rcx
  __int64 v45; // [rsp+10h] [rbp-120h]
  __int64 v46; // [rsp+18h] [rbp-118h]
  __int64 v47; // [rsp+28h] [rbp-108h] BYREF
  __int64 v48; // [rsp+30h] [rbp-100h] BYREF
  __int64 v49; // [rsp+38h] [rbp-F8h] BYREF
  __int64 v50; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v51; // [rsp+48h] [rbp-E8h]
  __int64 v52; // [rsp+50h] [rbp-E0h]
  __int64 v53; // [rsp+58h] [rbp-D8h]
  __int64 v54[2]; // [rsp+60h] [rbp-D0h] BYREF
  _QWORD *v55; // [rsp+70h] [rbp-C0h]
  __int64 v56; // [rsp+78h] [rbp-B8h]
  __int64 v57; // [rsp+80h] [rbp-B0h]
  __int64 v58; // [rsp+88h] [rbp-A8h]
  _QWORD *v59; // [rsp+90h] [rbp-A0h]
  _QWORD *v60; // [rsp+98h] [rbp-98h]
  __int64 v61; // [rsp+A0h] [rbp-90h]
  _QWORD *v62; // [rsp+A8h] [rbp-88h]
  __int64 *v63; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v64; // [rsp+B8h] [rbp-78h]
  __int64 *v65; // [rsp+C0h] [rbp-70h]
  __int64 v66; // [rsp+C8h] [rbp-68h]
  __int64 v67; // [rsp+D0h] [rbp-60h]
  __int64 v68; // [rsp+D8h] [rbp-58h]
  __int64 *v69; // [rsp+E0h] [rbp-50h]
  __int64 *v70; // [rsp+E8h] [rbp-48h]
  __int64 v71; // [rsp+F0h] [rbp-40h]
  _QWORD *v72; // [rsp+F8h] [rbp-38h]

  v4 = a2;
  v47 = a1;
  v54[0] = 0;
  v54[1] = 0;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  sub_1C08D60(v54, 0);
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  if ( sub_183E920(a2, a1) )
    goto LABEL_50;
  v5 = v59;
  if ( v59 == (_QWORD *)(v61 - 8) )
  {
    sub_1B4ECC0(v54, &v47);
    v6 = (__int64)v59;
  }
  else
  {
    if ( v59 )
    {
      *v59 = v47;
      v5 = v59;
    }
    v6 = (__int64)(v5 + 1);
    v59 = (_QWORD *)v6;
  }
  if ( v55 == (_QWORD *)v6 )
    goto LABEL_50;
  do
  {
    while ( 1 )
    {
      if ( v60 == (_QWORD *)v6 )
        v6 = *(v62 - 1) + 512LL;
      v48 = *(_QWORD *)(v6 - 8);
      v13 = sub_183E920(v4, v48);
      if ( v13 )
        break;
      v14 = v48;
      v15 = *(__int64 **)(v4 + 8);
      if ( *(__int64 **)(v4 + 16) != v15 )
        goto LABEL_20;
      v36 = &v15[*(unsigned int *)(v4 + 28)];
      v37 = *(_DWORD *)(v4 + 28);
      if ( v15 == v36 )
      {
LABEL_73:
        if ( v37 >= *(_DWORD *)(v4 + 24) )
        {
LABEL_20:
          sub_16CCBA0(v4, v48);
          goto LABEL_21;
        }
        *(_DWORD *)(v4 + 28) = v37 + 1;
        *v36 = v14;
        ++*(_QWORD *)v4;
      }
      else
      {
        v38 = 0;
        while ( v48 != *v15 )
        {
          if ( *v15 == -2 )
            v38 = v15;
          if ( v36 == ++v15 )
          {
            if ( !v38 )
              goto LABEL_73;
            *v38 = v48;
            v16 = v53;
            --*(_DWORD *)(v4 + 32);
            ++*(_QWORD *)v4;
            if ( v16 )
              goto LABEL_22;
            goto LABEL_64;
          }
        }
      }
LABEL_21:
      v16 = v53;
      if ( !(_DWORD)v53 )
      {
LABEL_64:
        ++v50;
LABEL_65:
        sub_13B3D40((__int64)&v50, 2 * v16);
        goto LABEL_66;
      }
LABEL_22:
      v17 = v48;
      v18 = (v16 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
      v19 = (__int64 *)(v51 + 8LL * v18);
      v20 = *v19;
      if ( v48 == *v19 )
        goto LABEL_23;
      v43 = 1;
      v39 = 0;
      while ( v20 != -8 )
      {
        if ( v39 || v20 != -16 )
          v19 = v39;
        v18 = (v16 - 1) & (v43 + v18);
        v44 = (__int64 *)(v51 + 8LL * v18);
        v20 = *v44;
        if ( v48 == *v44 )
          goto LABEL_23;
        ++v43;
        v39 = v19;
        v19 = (__int64 *)(v51 + 8LL * v18);
      }
      if ( !v39 )
        v39 = v19;
      ++v50;
      v40 = v52 + 1;
      if ( 4 * ((int)v52 + 1) >= 3 * v16 )
        goto LABEL_65;
      if ( v16 - HIDWORD(v52) - v40 > v16 >> 3 )
        goto LABEL_81;
      sub_13B3D40((__int64)&v50, v16);
LABEL_66:
      sub_1898220((__int64)&v50, &v48, &v63);
      v39 = v63;
      v17 = v48;
      v40 = v52 + 1;
LABEL_81:
      LODWORD(v52) = v40;
      if ( *v39 != -8 )
        --HIDWORD(v52);
      *v39 = v17;
LABEL_23:
      v63 = 0;
      v64 = 0;
      v65 = 0;
      v66 = 0;
      v67 = 0;
      v68 = 0;
      v69 = 0;
      v70 = 0;
      v71 = 0;
      v72 = 0;
      sub_1C08D60((__int64 *)&v63, 0);
      v21 = sub_157EBA0(v48);
      v22 = v21;
      if ( v21 )
      {
        v23 = sub_15F4D60(v21);
        if ( v23 )
        {
          v46 = v4;
          v45 = a3;
          do
          {
            while ( 1 )
            {
              v25 = sub_15F4DF0(v22, v13);
              v24 = v69;
              if ( v69 != (__int64 *)(v71 - 8) )
                break;
              v26 = v72;
              if ( v69 - v70 + (((((__int64)v72 - v68) >> 3) - 1) << 6) + ((v67 - (__int64)v65) >> 3) == 0xFFFFFFFFFFFFFFFLL )
                sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
              if ( (unsigned __int64)(v64 - (v72 - v63)) <= 1 )
              {
                sub_1C09D80((__int64 *)&v63, 1u, 0);
                v26 = v72;
              }
              v26[1] = sub_22077B0(512);
              if ( v69 )
                *v69 = v25;
              ++v13;
              v27 = *++v72 + 512LL;
              v70 = (__int64 *)*v72;
              v71 = v27;
              v69 = v70;
              if ( v23 == v13 )
                goto LABEL_36;
            }
            if ( v69 )
            {
              *v69 = v25;
              v24 = v69;
            }
            ++v13;
            v69 = v24 + 1;
          }
          while ( v23 != v13 );
LABEL_36:
          v4 = v46;
          a3 = v45;
        }
      }
      v28 = v69;
      if ( v65 != v69 )
      {
        while ( 1 )
        {
          if ( v28 == v70 )
          {
            v49 = *(_QWORD *)(*(v72 - 1) + 504LL);
            j_j___libc_free_0(v28, 512);
            v31 = v49;
            v34 = *--v72 + 512LL;
            v70 = (__int64 *)*v72;
            v71 = v34;
            v69 = v70 + 63;
          }
          else
          {
            v31 = *(v28 - 1);
            v69 = v28 - 1;
            v49 = v31;
          }
          if ( !(_DWORD)v53 )
            goto LABEL_44;
          v29 = (v53 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
          v30 = *(_QWORD *)(v51 + 8LL * v29);
          if ( v31 == v30 )
          {
LABEL_40:
            v28 = v69;
            if ( v69 == v65 )
              break;
          }
          else
          {
            v35 = 1;
            while ( v30 != -8 )
            {
              v29 = (v53 - 1) & (v35 + v29);
              v30 = *(_QWORD *)(v51 + 8LL * v29);
              if ( v30 == v31 )
                goto LABEL_40;
              ++v35;
            }
LABEL_44:
            if ( sub_183E920(v4, v31) )
              goto LABEL_40;
            v32 = v59;
            if ( v59 == (_QWORD *)(v61 - 8) )
            {
              sub_1B4ECC0(v54, &v49);
              goto LABEL_40;
            }
            if ( v59 )
            {
              *v59 = v49;
              v32 = v59;
            }
            v28 = v69;
            v59 = v32 + 1;
            if ( v69 == v65 )
              break;
          }
        }
      }
      sub_1C08CE0((__int64 *)&v63);
      v6 = (__int64)v59;
      if ( v59 == v55 )
        goto LABEL_50;
    }
    if ( v59 == v60 )
    {
      j_j___libc_free_0(v59, 512);
      v41 = *--v62 + 512LL;
      v60 = (_QWORD *)*v62;
      v61 = v41;
      v59 = v60 + 63;
    }
    else
    {
      --v59;
    }
    if ( (_DWORD)v53 )
    {
      v7 = v51;
      v8 = (v53 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
      v9 = (__int64 *)(v51 + 8LL * v8);
      v10 = *v9;
      if ( *v9 == v48 )
      {
LABEL_12:
        *v9 = -16;
        LODWORD(v52) = v52 - 1;
        ++HIDWORD(v52);
      }
      else
      {
        v42 = 1;
        while ( v10 != -8 )
        {
          v12 = v42 + 1;
          v8 = (v53 - 1) & (v42 + v8);
          v9 = (__int64 *)(v51 + 8LL * v8);
          v10 = *v9;
          if ( v48 == *v9 )
            goto LABEL_12;
          v42 = v12;
        }
      }
    }
    v11 = *(unsigned int *)(a3 + 8);
    if ( (unsigned int)v11 >= *(_DWORD *)(a3 + 12) )
    {
      sub_16CD150(a3, (const void *)(a3 + 16), 0, 8, v7, v12);
      v11 = *(unsigned int *)(a3 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a3 + 8 * v11) = v48;
    v6 = (__int64)v59;
    ++*(_DWORD *)(a3 + 8);
  }
  while ( (_QWORD *)v6 != v55 );
LABEL_50:
  j___libc_free_0(v51);
  return sub_1C08CE0(v54);
}
