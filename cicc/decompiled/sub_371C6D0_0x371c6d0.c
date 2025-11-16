// Function: sub_371C6D0
// Address: 0x371c6d0
//
__int64 __fastcall sub_371C6D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v6; // rsi
  __int64 v7; // rbx
  __int64 v8; // rax
  unsigned int v9; // r8d
  _QWORD *v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // rbx
  __int64 v13; // rax
  unsigned __int8 v14; // dl
  __int64 *v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rsi
  _DWORD *v19; // rax
  __int64 v20; // rax
  __int64 v21; // r13
  __int64 v22; // r12
  __int64 v23; // rdi
  __int64 v24; // rsi
  int v26; // ebx
  int v27; // edx
  __int64 v28; // rax
  __int64 v29; // r8
  unsigned __int64 v30; // rbx
  __int64 v31; // rdx
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rsi
  unsigned __int64 *v34; // rcx
  __int64 v35; // r9
  int v36; // eax
  unsigned __int64 *v37; // rdx
  __int64 v38; // rax
  unsigned int v39; // ecx
  __int64 v40; // rdi
  __int64 *v41; // r10
  int v42; // r11d
  __int64 *v43; // r9
  unsigned int v44; // ecx
  __int64 v45; // rdi
  int v46; // r11d
  char *v47; // [rsp+8h] [rbp-C8h]
  __int64 v48; // [rsp+8h] [rbp-C8h]
  __int64 v50; // [rsp+18h] [rbp-B8h]
  __int64 **v52; // [rsp+30h] [rbp-A0h]
  __int64 v53; // [rsp+38h] [rbp-98h]
  _QWORD *v54; // [rsp+40h] [rbp-90h]
  __int64 *v55; // [rsp+40h] [rbp-90h]
  __int64 v56; // [rsp+40h] [rbp-90h]
  unsigned int v57; // [rsp+40h] [rbp-90h]
  __int64 v58; // [rsp+48h] [rbp-88h]
  unsigned __int64 v59; // [rsp+58h] [rbp-78h] BYREF
  __int64 v60; // [rsp+60h] [rbp-70h] BYREF
  __int64 v61; // [rsp+68h] [rbp-68h]
  __int64 v62; // [rsp+70h] [rbp-60h]
  unsigned int v63; // [rsp+78h] [rbp-58h]
  char v64[8]; // [rsp+80h] [rbp-50h] BYREF
  __int64 v65; // [rsp+88h] [rbp-48h]

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x600000000LL;
  v52 = *(__int64 ***)(a2 + 24);
  v3 = *(_QWORD *)(a2 + 16);
  v4 = *(_QWORD *)(v3 + 80);
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v53 = v4;
  v50 = v3 + 72;
  if ( v4 == v3 + 72 )
  {
    v23 = 0;
    v24 = 0;
    goto LABEL_29;
  }
  do
  {
    v6 = v53 - 24;
    if ( !v53 )
      v6 = 0;
    v7 = sub_3186770((__int64)v52, v6);
    sub_371B570((__int64)v64, v7);
    v58 = *(_QWORD *)(v7 + 16) + 48LL;
    if ( v58 != v65 )
    {
      while ( 1 )
      {
        v20 = sub_371B3B0((__int64)v64, v65);
        v21 = *(_QWORD *)(v20 + 16);
        v22 = v20;
        if ( *(_QWORD *)(v21 + 48) || (*(_BYTE *)(v21 + 7) & 0x20) != 0 )
        {
          v8 = sub_B91F50(*(_QWORD *)(v20 + 16), "sandboxvec", 0xAu);
          if ( v8 )
            break;
        }
LABEL_19:
        sub_371B2F0((__int64)v64);
        if ( v65 == v58 )
          goto LABEL_27;
      }
      if ( v63 )
      {
        v9 = (v63 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v10 = (_QWORD *)(v61 + 16LL * v9);
        v11 = *v10;
        if ( v8 == *v10 )
        {
LABEL_9:
          v12 = v10[1];
LABEL_10:
          sub_371C3E0(v12, v22);
          if ( *(_QWORD *)(v21 + 48) || (*(_BYTE *)(v21 + 7) & 0x20) != 0 )
          {
            v13 = sub_B91F50(v21, "sandboxaux", 0xAu);
            if ( v13 )
            {
              v14 = *(_BYTE *)(v13 - 16);
              if ( (v14 & 2) != 0 )
                v15 = *(__int64 **)(v13 - 32);
              else
                v15 = (__int64 *)(-16 - 8LL * ((v14 >> 2) & 0xF) + v13);
              v16 = *v15;
              if ( *(_BYTE *)v16 != 1 )
                BUG();
              v17 = *(_QWORD *)(v16 + 136);
              LODWORD(v18) = *(_DWORD *)(v17 + 32);
              v19 = *(_DWORD **)(v17 + 24);
              if ( (unsigned int)v18 > 0x40 )
              {
                LODWORD(v18) = *v19;
              }
              else if ( (_DWORD)v18 )
              {
                v18 = (__int64)((_QWORD)v19 << (64 - (unsigned __int8)v18)) >> (64 - (unsigned __int8)v18);
              }
              sub_371BF00(v12, v18, v22);
            }
          }
          goto LABEL_19;
        }
        v54 = 0;
        v26 = 1;
        while ( v11 != -4096 )
        {
          if ( !v54 )
          {
            if ( v11 != -8192 )
              v10 = 0;
            v54 = v10;
          }
          v9 = (v63 - 1) & (v26 + v9);
          v10 = (_QWORD *)(v61 + 16LL * v9);
          v11 = *v10;
          if ( v8 == *v10 )
            goto LABEL_9;
          ++v26;
        }
        if ( v54 )
          v10 = v54;
        ++v60;
        v55 = v10;
        v27 = v62 + 1;
        if ( 4 * ((int)v62 + 1) < 3 * v63 )
        {
          if ( v63 - HIDWORD(v62) - v27 > v63 >> 3 )
            goto LABEL_36;
          v57 = ((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4);
          v48 = v8;
          sub_371C200((__int64)&v60, v63);
          if ( !v63 )
          {
LABEL_81:
            LODWORD(v62) = v62 + 1;
            BUG();
          }
          v44 = (v63 - 1) & v57;
          v55 = (__int64 *)(v61 + 16LL * v44);
          v45 = *v55;
          v27 = v62 + 1;
          v8 = v48;
          if ( v48 == *v55 )
            goto LABEL_36;
          v41 = (__int64 *)(v61 + 16LL * v44);
          v46 = 1;
          v43 = 0;
          while ( v45 != -4096 )
          {
            if ( !v43 && v45 == -8192 )
              v43 = v41;
            v44 = (v63 - 1) & (v46 + v44);
            v41 = (__int64 *)(v61 + 16LL * v44);
            v45 = *v41;
            if ( v48 == *v41 )
              goto LABEL_66;
            ++v46;
          }
          goto LABEL_55;
        }
      }
      else
      {
        ++v60;
      }
      v56 = v8;
      sub_371C200((__int64)&v60, 2 * v63);
      if ( !v63 )
        goto LABEL_81;
      v8 = v56;
      v39 = (v63 - 1) & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
      v55 = (__int64 *)(v61 + 16LL * v39);
      v40 = *v55;
      v27 = v62 + 1;
      if ( v8 == *v55 )
        goto LABEL_36;
      v41 = (__int64 *)(v61 + 16LL * v39);
      v42 = 1;
      v43 = 0;
      while ( v40 != -4096 )
      {
        if ( !v43 && v40 == -8192 )
          v43 = v41;
        v39 = (v63 - 1) & (v42 + v39);
        v41 = (__int64 *)(v61 + 16LL * v39);
        v40 = *v41;
        if ( v8 == *v41 )
        {
LABEL_66:
          v55 = v41;
          goto LABEL_36;
        }
        ++v42;
      }
LABEL_55:
      if ( !v43 )
        v43 = v41;
      v55 = v43;
LABEL_36:
      LODWORD(v62) = v27;
      if ( *v55 != -4096 )
        --HIDWORD(v62);
      *v55 = v8;
      v55[1] = 0;
      v28 = sub_22077B0(0xC0u);
      v30 = v28;
      if ( v28 )
        sub_371BA10(v28, v52, a3);
      v31 = *(unsigned int *)(a1 + 8);
      v32 = *(unsigned int *)(a1 + 12);
      v59 = v30;
      v33 = *(_QWORD *)a1;
      v34 = &v59;
      v35 = v31 + 1;
      v36 = v31;
      if ( v31 + 1 > v32 )
      {
        if ( v33 > (unsigned __int64)&v59 || (unsigned __int64)&v59 >= v33 + 8 * v31 )
        {
          sub_31B5960(a1, v31 + 1, v31, (__int64)&v59, v29, v35);
          v31 = *(unsigned int *)(a1 + 8);
          v33 = *(_QWORD *)a1;
          v34 = &v59;
          v36 = *(_DWORD *)(a1 + 8);
        }
        else
        {
          v47 = (char *)&v59 - v33;
          sub_31B5960(a1, v31 + 1, v31, (__int64)&v59 - v33, v29, v35);
          v33 = *(_QWORD *)a1;
          v31 = *(unsigned int *)(a1 + 8);
          v34 = (unsigned __int64 *)&v47[*(_QWORD *)a1];
          v36 = *(_DWORD *)(a1 + 8);
        }
      }
      v37 = (unsigned __int64 *)(v33 + 8 * v31);
      if ( v37 )
      {
        *v37 = *v34;
        *v34 = 0;
        v30 = v59;
        v36 = *(_DWORD *)(a1 + 8);
      }
      v38 = (unsigned int)(v36 + 1);
      *(_DWORD *)(a1 + 8) = v38;
      if ( v30 )
      {
        sub_371BB90(v30);
        j_j___libc_free_0(v30);
        v38 = *(unsigned int *)(a1 + 8);
      }
      v12 = *(_QWORD *)(*(_QWORD *)a1 + 8 * v38 - 8);
      v55[1] = v12;
      goto LABEL_10;
    }
LABEL_27:
    v53 = *(_QWORD *)(v53 + 8);
  }
  while ( v50 != v53 );
  v23 = v61;
  v24 = 16LL * v63;
LABEL_29:
  sub_C7D6A0(v23, v24, 8);
  return a1;
}
