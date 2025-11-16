// Function: sub_2909030
// Address: 0x2909030
//
void __fastcall sub_2909030(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v4; // r13
  __int64 v7; // rdi
  unsigned __int64 v8; // rdx
  __int16 v9; // dx
  __int64 v10; // r15
  unsigned __int8 v11; // al
  char v12; // dl
  __int64 v13; // rcx
  __int64 v14; // rax
  __int16 v15; // dx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rcx
  __int64 v19; // r10
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 *v22; // rax
  unsigned __int8 *v23; // r15
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 *v28; // rax
  unsigned __int8 *v29; // rax
  unsigned __int8 *v30; // rdx
  _QWORD *v31; // rax
  __int64 v32; // r15
  _QWORD *v33; // rax
  __int64 v34; // r15
  __int64 v35; // rdx
  __int64 v36; // rsi
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rbx
  signed __int64 v40; // rax
  int v41; // edx
  _BYTE *v42; // r14
  _BYTE *v43; // r12
  __int64 v44; // rdx
  int v45; // eax
  __int64 v46; // rdi
  int v47; // ecx
  unsigned int v48; // eax
  __int64 *v49; // rsi
  __int64 v50; // r8
  __int64 v51; // rax
  _QWORD *v52; // rdi
  __int64 v53; // rsi
  _QWORD *v54; // rax
  int v55; // r8d
  __int64 v56; // r15
  __int64 v57; // r15
  __int64 *v58; // rax
  __int64 v59; // rcx
  unsigned __int8 *v60; // rax
  int v61; // esi
  int v62; // r9d
  int v63; // eax
  int v64; // r10d
  __int64 v65; // rcx
  __int64 v66; // r8
  __int64 v67; // r9
  bool v68; // cc
  __int64 v70; // [rsp+10h] [rbp-1B0h]
  __int64 v71; // [rsp+18h] [rbp-1A8h]
  __int64 v72; // [rsp+20h] [rbp-1A0h]
  unsigned __int8 *v73; // [rsp+20h] [rbp-1A0h]
  unsigned __int8 *v74; // [rsp+20h] [rbp-1A0h]
  __int64 v75; // [rsp+28h] [rbp-198h]
  __int64 v76; // [rsp+28h] [rbp-198h]
  __int64 v77; // [rsp+28h] [rbp-198h]
  __int64 v78; // [rsp+30h] [rbp-190h]
  __int64 *v80; // [rsp+48h] [rbp-178h]
  __int64 v81; // [rsp+58h] [rbp-168h] BYREF
  __int64 v82; // [rsp+60h] [rbp-160h] BYREF
  __int64 v83; // [rsp+68h] [rbp-158h]
  unsigned __int8 *v84; // [rsp+70h] [rbp-150h]
  _BYTE *v85; // [rsp+80h] [rbp-140h] BYREF
  __int64 v86; // [rsp+88h] [rbp-138h]
  _BYTE v87[304]; // [rsp+90h] [rbp-130h] BYREF

  v4 = *(__int64 **)(a2 + 32);
  v85 = v87;
  v86 = 0x2000000000LL;
  v80 = &v4[*(unsigned int *)(a2 + 40)];
  if ( v4 == v80 )
    return;
  v7 = 0;
  v78 = a2 + 64;
  do
  {
    while ( 1 )
    {
      v34 = *v4;
      v35 = *(unsigned int *)(a4 + 24);
      v36 = *(_QWORD *)(a4 + 8);
      v81 = *v4;
      if ( !(_DWORD)v35 )
        goto LABEL_18;
      v67 = (unsigned int)(v35 - 1);
      v65 = (unsigned int)v67 & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
      v37 = v36 + 16 * v65;
      v66 = *(_QWORD *)v37;
      if ( v34 != *(_QWORD *)v37 )
      {
        v63 = 1;
        while ( v66 != -4096 )
        {
          v64 = v63 + 1;
          v65 = (unsigned int)v67 & (v63 + (_DWORD)v65);
          v37 = v36 + 16LL * (unsigned int)v65;
          v66 = *(_QWORD *)v37;
          if ( v34 == *(_QWORD *)v37 )
            goto LABEL_21;
          v63 = v64;
        }
        goto LABEL_18;
      }
LABEL_21:
      if ( v37 == v36 + 16 * v35 )
        goto LABEL_18;
      v38 = *(_QWORD *)(a4 + 32);
      v39 = v38 + 72LL * *(unsigned int *)(v37 + 8);
      if ( v39 == v38 + 72LL * *(unsigned int *)(a4 + 40) )
        goto LABEL_18;
      v40 = *(_QWORD *)(v39 + 56);
      v41 = *(_DWORD *)(v39 + 64);
      if ( *(_BYTE *)a1 == 34 )
      {
        v65 = 2 * v40;
        if ( is_mul_ok(2u, v40) )
        {
          v40 *= 2LL;
        }
        else
        {
          v65 = 0x8000000000000000LL;
          v68 = v40 <= 0;
          v40 = 0x7FFFFFFFFFFFFFFFLL;
          if ( v68 )
            v40 = 0x8000000000000000LL;
        }
      }
      if ( !v41 )
        break;
      if ( v41 < 0 )
        goto LABEL_4;
LABEL_18:
      if ( v80 == ++v4 )
        goto LABEL_27;
    }
    if ( (unsigned int)qword_50050A8 > v40 )
    {
LABEL_4:
      v8 = v7 + 1;
      if ( v7 + 1 > (unsigned __int64)HIDWORD(v86) )
      {
        sub_C8D5F0((__int64)&v85, v87, v8, 8u, v66, v67);
        v7 = (unsigned int)v86;
      }
      *(_QWORD *)&v85[8 * v7] = v34;
      LODWORD(v86) = v86 + 1;
      if ( *(_BYTE *)a1 == 85 )
      {
        v56 = *(_QWORD *)(a1 + 32);
        if ( v56 == *(_QWORD *)(a1 + 40) + 48LL || !v56 )
          v57 = 0;
        else
          v57 = v56 - 24;
        v58 = (__int64 *)sub_1152A40(a3, &v81, v8, v65, v66, v67);
        v59 = v70;
        LOWORD(v59) = 0;
        v70 = v59;
        v60 = sub_28FEA70(*(_QWORD *)(v39 + 8), *(unsigned int *)(v39 + 16), v57 + 24, v59, *(_QWORD *)(v39 + 48), *v58);
        v32 = v81;
        v82 = 0;
        v84 = v60;
        v83 = 0;
        if ( v60 + 4096 == 0 || v60 == 0 || v60 == (unsigned __int8 *)-8192LL )
          goto LABEL_17;
      }
      else
      {
        v10 = sub_AA5190(*(_QWORD *)(a1 - 96));
        if ( v10 )
        {
          v11 = v9;
          v12 = HIBYTE(v9);
        }
        else
        {
          v12 = 0;
          v11 = 0;
        }
        v13 = v11;
        BYTE1(v13) = v12;
        v75 = v13;
        v14 = sub_AA5190(*(_QWORD *)(a1 - 64));
        v18 = v75;
        v19 = v14;
        if ( v14 )
        {
          LOBYTE(v20) = v15;
          v21 = HIBYTE(v15);
        }
        else
        {
          v21 = 0;
          LOBYTE(v20) = 0;
        }
        v72 = v19;
        v71 = v75;
        v20 = (unsigned __int8)v20;
        BYTE1(v20) = v21;
        v76 = v20;
        v22 = (__int64 *)sub_1152A40(a3, &v81, v21, v18, v16, v17);
        v23 = sub_28FEA70(*(_QWORD *)(v39 + 8), *(unsigned int *)(v39 + 16), v10, v71, *(_QWORD *)(v39 + 48), *v22);
        v28 = (__int64 *)sub_1152A40(a3, &v81, v24, v25, v26, v27);
        v29 = sub_28FEA70(*(_QWORD *)(v39 + 8), *(unsigned int *)(v39 + 16), v72, v76, *(_QWORD *)(v39 + 48), *v28);
        v84 = v23;
        v30 = v29;
        v82 = 0;
        v77 = v81;
        v83 = 0;
        if ( v23 + 4096 != 0 && v23 != 0 && v23 != (unsigned __int8 *)-8192LL )
        {
          v73 = v29;
          sub_BD73F0((__int64)&v82);
          v30 = v73;
        }
        v74 = v30;
        v31 = (_QWORD *)sub_2908C80(v78, &v82);
        sub_FC7530(v31, v77);
        sub_D68D70(&v82);
        v82 = 0;
        v83 = 0;
        v32 = v81;
        v84 = v74;
        if ( v74 == 0 || v74 + 4096 == 0 || v74 == (unsigned __int8 *)-8192LL )
          goto LABEL_17;
      }
      sub_BD73F0((__int64)&v82);
LABEL_17:
      v33 = (_QWORD *)sub_2908C80(v78, &v82);
      sub_FC7530(v33, v32);
      sub_D68D70(&v82);
      v7 = (unsigned int)v86;
      goto LABEL_18;
    }
    ++v4;
  }
  while ( v80 != v4 );
LABEL_27:
  v42 = v85;
  v43 = &v85[8 * v7];
  if ( v85 != v43 )
  {
    do
    {
      v44 = *(_QWORD *)v42;
      v45 = *(_DWORD *)(a2 + 24);
      v46 = *(_QWORD *)(a2 + 8);
      v82 = *(_QWORD *)v42;
      if ( v45 )
      {
        v47 = v45 - 1;
        v48 = (v45 - 1) & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
        v49 = (__int64 *)(v46 + 8LL * v48);
        v50 = *v49;
        if ( v44 == *v49 )
        {
LABEL_30:
          *v49 = -8192;
          v51 = *(unsigned int *)(a2 + 40);
          --*(_DWORD *)(a2 + 16);
          v52 = *(_QWORD **)(a2 + 32);
          ++*(_DWORD *)(a2 + 20);
          v53 = (__int64)&v52[v51];
          v54 = sub_28FEBC0(v52, v53, &v82);
          if ( v54 + 1 != (_QWORD *)v53 )
          {
            memmove(v54, v54 + 1, v53 - (_QWORD)(v54 + 1));
            v55 = *(_DWORD *)(a2 + 40);
          }
          *(_DWORD *)(a2 + 40) = v55 - 1;
        }
        else
        {
          v61 = 1;
          while ( v50 != -4096 )
          {
            v62 = v61 + 1;
            v48 = v47 & (v61 + v48);
            v49 = (__int64 *)(v46 + 8LL * v48);
            v50 = *v49;
            if ( v44 == *v49 )
              goto LABEL_30;
            v61 = v62;
          }
        }
      }
      v42 += 8;
    }
    while ( v43 != v42 );
    v42 = v85;
  }
  if ( v42 != v87 )
    _libc_free((unsigned __int64)v42);
}
