// Function: sub_2C9D870
// Address: 0x2c9d870
//
void __fastcall sub_2C9D870(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r8
  __int64 v6; // r9
  bool v7; // zf
  _QWORD *v8; // rax
  _QWORD *v9; // rdx
  __int64 v10; // rdi
  __int64 v11; // rsi
  _QWORD *v12; // rax
  __int64 v13; // rax
  unsigned __int64 v14; // rdi
  __int64 v15; // rsi
  _QWORD *v16; // rdx
  _QWORD *v17; // rcx
  _QWORD *v18; // rax
  _QWORD *v19; // rax
  __int64 v20; // r12
  unsigned int v21; // eax
  __int64 *v22; // rcx
  __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 *v26; // rax
  unsigned int v27; // esi
  __int64 v28; // rdx
  unsigned int v29; // eax
  __int64 *v30; // rdi
  __int64 v31; // rcx
  unsigned __int64 v32; // rax
  __int64 v33; // r13
  __int64 *v34; // r15
  unsigned int i; // r12d
  __int64 v36; // r9
  __int64 **v37; // rsi
  __int64 v38; // rax
  __int64 v39; // rsi
  unsigned int v40; // eax
  __int64 v41; // rdi
  __int64 v42; // rdx
  int v43; // r8d
  _QWORD *v44; // rax
  _QWORD *v45; // rdx
  _QWORD *v46; // rax
  __int64 v47; // rcx
  __int64 *v48; // rdx
  __int64 *v49; // r10
  int v50; // ecx
  int v51; // ecx
  int v52; // r11d
  __int64 *v53; // r12
  __int64 v54; // [rsp+18h] [rbp-128h]
  __int64 v55; // [rsp+20h] [rbp-120h]
  int v56; // [rsp+34h] [rbp-10Ch]
  __int64 v57; // [rsp+38h] [rbp-108h] BYREF
  __int64 v58; // [rsp+40h] [rbp-100h] BYREF
  __int64 v59; // [rsp+48h] [rbp-F8h] BYREF
  __int64 v60; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v61; // [rsp+58h] [rbp-E8h]
  __int64 v62; // [rsp+60h] [rbp-E0h]
  __int64 v63; // [rsp+68h] [rbp-D8h]
  __int64 v64[2]; // [rsp+70h] [rbp-D0h] BYREF
  _QWORD *v65; // [rsp+80h] [rbp-C0h]
  __int64 v66; // [rsp+88h] [rbp-B8h]
  __int64 v67; // [rsp+90h] [rbp-B0h]
  __int64 v68; // [rsp+98h] [rbp-A8h]
  _QWORD *v69; // [rsp+A0h] [rbp-A0h]
  _QWORD *v70; // [rsp+A8h] [rbp-98h]
  __int64 v71; // [rsp+B0h] [rbp-90h]
  _QWORD *v72; // [rsp+B8h] [rbp-88h]
  __int64 *v73; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v74; // [rsp+C8h] [rbp-78h]
  __int64 *v75; // [rsp+D0h] [rbp-70h]
  __int64 v76; // [rsp+D8h] [rbp-68h]
  __int64 v77; // [rsp+E0h] [rbp-60h]
  __int64 v78; // [rsp+E8h] [rbp-58h]
  __int64 *v79; // [rsp+F0h] [rbp-50h]
  __int64 *v80; // [rsp+F8h] [rbp-48h]
  __int64 v81; // [rsp+100h] [rbp-40h]
  __int64 **v82; // [rsp+108h] [rbp-38h]

  v57 = a1;
  v64[0] = 0;
  v64[1] = 0;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v68 = 0;
  v69 = 0;
  v70 = 0;
  v71 = 0;
  v72 = 0;
  sub_2768EA0(v64, 0);
  v7 = *(_BYTE *)(a2 + 28) == 0;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  if ( !v7 )
  {
    v8 = *(_QWORD **)(a2 + 8);
    v9 = &v8[*(unsigned int *)(a2 + 20)];
    if ( v8 == v9 )
      goto LABEL_9;
    while ( v57 != *v8 )
    {
      if ( v9 == ++v8 )
        goto LABEL_9;
    }
    v10 = 0;
    v11 = 0;
LABEL_7:
    sub_C7D6A0(v10, v11, 8);
    sub_2767770((unsigned __int64 *)v64);
    return;
  }
  if ( sub_C8CA60(a2, a1) )
  {
    v10 = v61;
    v11 = 8LL * (unsigned int)v63;
    goto LABEL_7;
  }
LABEL_9:
  v12 = v69;
  if ( v69 == (_QWORD *)(v71 - 8) )
  {
    sub_27698B0((unsigned __int64 *)v64, &v57);
    v13 = (__int64)v69;
  }
  else
  {
    if ( v69 )
    {
      *v69 = v57;
      v12 = v69;
    }
    v13 = (__int64)(v12 + 1);
    v69 = (_QWORD *)v13;
  }
  if ( v65 != (_QWORD *)v13 )
  {
    do
    {
      v14 = (unsigned __int64)v70;
      if ( v70 == (_QWORD *)v13 )
        v13 = *(v72 - 1) + 512LL;
      v15 = *(_QWORD *)(v13 - 8);
      v7 = *(_BYTE *)(a2 + 28) == 0;
      v58 = v15;
      if ( v7 )
      {
        if ( sub_C8CA60(a2, v15) )
        {
          v14 = (unsigned __int64)v70;
          v19 = v69;
          if ( v69 == v70 )
            goto LABEL_33;
LABEL_22:
          v69 = v19 - 1;
LABEL_23:
          v20 = v58;
          if ( (_DWORD)v63 )
          {
            v21 = (v63 - 1) & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
            v22 = (__int64 *)(v61 + 8LL * v21);
            v23 = *v22;
            if ( *v22 == v58 )
            {
LABEL_25:
              *v22 = -8192;
              v20 = v58;
              LODWORD(v62) = v62 - 1;
              ++HIDWORD(v62);
            }
            else
            {
              v51 = 1;
              while ( v23 != -4096 )
              {
                v5 = (unsigned int)(v51 + 1);
                v21 = (v63 - 1) & (v51 + v21);
                v22 = (__int64 *)(v61 + 8LL * v21);
                v23 = *v22;
                if ( *v22 == v58 )
                  goto LABEL_25;
                v51 = v5;
              }
            }
          }
          v24 = *(unsigned int *)(a3 + 8);
          if ( v24 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
          {
            sub_C8D5F0(a3, (const void *)(a3 + 16), v24 + 1, 8u, v5, v6);
            v24 = *(unsigned int *)(a3 + 8);
          }
          *(_QWORD *)(*(_QWORD *)a3 + 8 * v24) = v20;
          ++*(_DWORD *)(a3 + 8);
          goto LABEL_29;
        }
        v15 = v58;
        if ( !*(_BYTE *)(a2 + 28) )
        {
LABEL_81:
          sub_C8CC70(a2, v15, (__int64)v48, v47, v5, v6);
          goto LABEL_37;
        }
        v26 = *(__int64 **)(a2 + 8);
        v47 = *(unsigned int *)(a2 + 20);
        v48 = &v26[v47];
        if ( v26 != v48 )
        {
LABEL_36:
          while ( *v26 != v15 )
          {
            if ( ++v26 == v48 )
              goto LABEL_75;
          }
LABEL_37:
          v27 = v63;
          if ( !(_DWORD)v63 )
            goto LABEL_77;
          goto LABEL_38;
        }
      }
      else
      {
        v16 = *(_QWORD **)(a2 + 8);
        v17 = &v16[*(unsigned int *)(a2 + 20)];
        v18 = v16;
        if ( v16 != v17 )
        {
          while ( v15 != *v18 )
          {
            if ( v17 == ++v18 )
            {
              v26 = *(__int64 **)(a2 + 8);
              v47 = *(unsigned int *)(a2 + 20);
              v48 = &v16[v47];
              goto LABEL_36;
            }
          }
          v19 = v69;
          if ( v69 != v70 )
            goto LABEL_22;
LABEL_33:
          j_j___libc_free_0(v14);
          v25 = *--v72 + 512LL;
          v70 = (_QWORD *)*v72;
          v71 = v25;
          v69 = v70 + 63;
          goto LABEL_23;
        }
        v47 = *(unsigned int *)(a2 + 20);
        v48 = &v16[v47];
      }
LABEL_75:
      if ( *(_DWORD *)(a2 + 16) <= (unsigned int)v47 )
        goto LABEL_81;
      *(_DWORD *)(a2 + 20) = v47 + 1;
      *v48 = v15;
      v27 = v63;
      ++*(_QWORD *)a2;
      if ( !v27 )
      {
LABEL_77:
        ++v60;
        v73 = 0;
LABEL_78:
        sub_CF28B0((__int64)&v60, 2 * v27);
        goto LABEL_79;
      }
LABEL_38:
      v28 = v58;
      v29 = (v27 - 1) & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
      v30 = (__int64 *)(v61 + 8LL * v29);
      v31 = *v30;
      if ( *v30 == v58 )
        goto LABEL_39;
      v52 = 1;
      v49 = 0;
      while ( v31 != -4096 )
      {
        if ( v49 || v31 != -8192 )
          v30 = v49;
        v29 = (v27 - 1) & (v52 + v29);
        v53 = (__int64 *)(v61 + 8LL * v29);
        v31 = *v53;
        if ( v58 == *v53 )
          goto LABEL_39;
        ++v52;
        v49 = v30;
        v30 = (__int64 *)(v61 + 8LL * v29);
      }
      if ( !v49 )
        v49 = v30;
      ++v60;
      v50 = v62 + 1;
      v73 = v49;
      if ( 4 * ((int)v62 + 1) >= 3 * v27 )
        goto LABEL_78;
      if ( v27 - HIDWORD(v62) - v50 > v27 >> 3 )
        goto LABEL_97;
      sub_CF28B0((__int64)&v60, v27);
LABEL_79:
      sub_D6B660((__int64)&v60, &v58, &v73);
      v28 = v58;
      v49 = v73;
      v50 = v62 + 1;
LABEL_97:
      LODWORD(v62) = v50;
      if ( *v49 != -4096 )
        --HIDWORD(v62);
      *v49 = v28;
LABEL_39:
      v73 = 0;
      v74 = 0;
      v75 = 0;
      v76 = 0;
      v77 = 0;
      v78 = 0;
      v79 = 0;
      v80 = 0;
      v81 = 0;
      v82 = 0;
      sub_2768EA0((__int64 *)&v73, 0);
      v32 = *(_QWORD *)(v58 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v32 == v58 + 48 )
        goto LABEL_82;
      if ( !v32 )
        BUG();
      v33 = v32 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v32 - 24) - 30 > 0xA )
      {
LABEL_82:
        v34 = v79;
      }
      else
      {
        v34 = v79;
        v56 = sub_B46E30(v33);
        if ( v56 )
        {
          for ( i = 0; i != v56; ++i )
          {
            while ( 1 )
            {
              v36 = sub_B46EC0(v33, i);
              if ( v34 != (__int64 *)(v81 - 8) )
                break;
              v37 = v82;
              if ( (((((__int64)v82 - v78) >> 3) - 1) << 6) + v34 - v80 + ((v77 - (__int64)v75) >> 3) == 0xFFFFFFFFFFFFFFFLL )
                sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
              if ( (unsigned __int64)(v74 - (((char *)v82 - (char *)v73) >> 3)) <= 1 )
              {
                v55 = v36;
                sub_27694A0((unsigned __int64 *)&v73, 1u, 0);
                v37 = v82;
                v36 = v55;
              }
              v54 = v36;
              v37[1] = (__int64 *)sub_22077B0(0x200u);
              if ( v79 )
                *v79 = v54;
              ++i;
              v34 = *++v82;
              v38 = (__int64)(*v82 + 64);
              v80 = v34;
              v81 = v38;
              v79 = v34;
              if ( v56 == i )
                goto LABEL_54;
            }
            if ( v34 )
            {
              *v34 = v36;
              v34 = v79;
            }
            v79 = ++v34;
          }
        }
      }
LABEL_54:
      if ( v75 != v34 )
      {
        while ( 1 )
        {
          if ( v34 == v80 )
          {
            v59 = (*(v82 - 1))[63];
            j_j___libc_free_0((unsigned __int64)v34);
            v39 = v59;
            v42 = (__int64)(*--v82 + 64);
            v80 = *v82;
            v81 = v42;
            v79 = v80 + 63;
          }
          else
          {
            v39 = *(v34 - 1);
            v79 = v34 - 1;
            v59 = v39;
          }
          if ( !(_DWORD)v63 )
            goto LABEL_64;
          v40 = (v63 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
          v41 = *(_QWORD *)(v61 + 8LL * v40);
          if ( v41 == v39 )
          {
LABEL_59:
            v34 = v79;
            if ( v79 == v75 )
              break;
          }
          else
          {
            v43 = 1;
            while ( v41 != -4096 )
            {
              v40 = (v63 - 1) & (v43 + v40);
              v41 = *(_QWORD *)(v61 + 8LL * v40);
              if ( v39 == v41 )
                goto LABEL_59;
              ++v43;
            }
LABEL_64:
            if ( *(_BYTE *)(a2 + 28) )
            {
              v44 = *(_QWORD **)(a2 + 8);
              v45 = &v44[*(unsigned int *)(a2 + 20)];
              if ( v44 != v45 )
              {
                while ( v39 != *v44 )
                {
                  if ( v45 == ++v44 )
                    goto LABEL_68;
                }
                goto LABEL_59;
              }
            }
            else if ( sub_C8CA60(a2, v39) )
            {
              goto LABEL_59;
            }
LABEL_68:
            v46 = v69;
            if ( v69 == (_QWORD *)(v71 - 8) )
            {
              sub_27698B0((unsigned __int64 *)v64, &v59);
              goto LABEL_59;
            }
            if ( v69 )
            {
              *v69 = v59;
              v46 = v69;
            }
            v34 = v79;
            v69 = v46 + 1;
            if ( v79 == v75 )
              break;
          }
        }
      }
      sub_2767770((unsigned __int64 *)&v73);
LABEL_29:
      v13 = (__int64)v69;
    }
    while ( v69 != v65 );
  }
  sub_C7D6A0(v61, 8LL * (unsigned int)v63, 8);
  sub_2767770((unsigned __int64 *)v64);
}
