// Function: sub_3572590
// Address: 0x3572590
//
__int64 __fastcall sub_3572590(unsigned __int8 *a1)
{
  __int64 v1; // rbx
  __int64 v2; // rbp
  unsigned __int8 v3; // dl
  unsigned __int64 v4; // rsi
  const char *v6; // r12
  __int64 result; // rax
  __int64 v8; // r12
  __int64 *v9; // rsi
  unsigned __int64 *v10; // rdi
  unsigned __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned int v14; // eax
  unsigned __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // r15
  __int64 v22; // r13
  unsigned __int64 v23; // rsi
  __int64 v24; // r13
  _QWORD *v25; // r14
  unsigned __int64 i; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  unsigned int v29; // eax
  __int64 v30; // rax
  __int64 v31; // rcx
  unsigned __int8 v32; // al
  unsigned int v33; // edx
  unsigned int v34; // eax
  __int64 v35; // r12
  size_t v36; // rsi
  __int64 v37; // rax
  __int64 v38; // rcx
  __int64 v39; // rsi
  unsigned __int64 v40; // rax
  unsigned __int64 v41; // rdx
  unsigned int v42; // eax
  _BYTE *v43; // r12
  __int64 v44; // rcx
  __int64 v45; // rsi
  unsigned __int64 v46; // rax
  unsigned __int64 v47; // rdx
  __int64 v48; // rax
  unsigned __int64 *v49; // rdi
  unsigned __int64 v50; // rsi
  __int64 v51; // rdi
  __int64 v52; // rax
  __int64 v53; // rdx
  unsigned int v54; // eax
  int *v55; // r12
  __int64 v56; // rax
  int *v57; // r13
  _BYTE *v58; // rdx
  _BYTE *v59; // rsi
  unsigned __int64 v60; // rax
  __int64 v61; // rax
  _QWORD *v62; // rdi
  unsigned __int64 v63; // rsi
  __int64 v64; // rax
  __int64 v65; // rdx
  unsigned __int64 v66; // r13
  size_t v67; // rax
  __int64 v68; // rax
  __int64 v69; // rdx
  const char *v70; // rax
  unsigned __int64 v71; // rdx
  __int64 v72; // rdx
  __int64 v73; // r9
  __int64 v74; // rax
  __int64 v75; // rbx
  __int64 v76; // rax
  unsigned int v77; // edx
  __int64 v78; // rsi
  __int64 v79; // r14
  __int64 v80; // rdx
  unsigned __int64 *v81; // rdi
  unsigned __int64 v82; // rsi
  unsigned int v83; // eax
  unsigned int v84; // eax
  __int64 v85; // [rsp-A0h] [rbp-A0h]
  __int64 v86; // [rsp-A0h] [rbp-A0h]
  __int64 v87; // [rsp-A0h] [rbp-A0h]
  __int64 v88; // [rsp-A0h] [rbp-A0h]
  unsigned __int64 v89; // [rsp-98h] [rbp-98h] BYREF
  _BYTE *v90; // [rsp-90h] [rbp-90h]
  _BYTE *v91; // [rsp-88h] [rbp-88h]
  unsigned __int64 v92; // [rsp-78h] [rbp-78h] BYREF
  __int64 v93; // [rsp-70h] [rbp-70h]
  unsigned __int64 v94; // [rsp-68h] [rbp-68h] BYREF
  _BOOL8 v95; // [rsp-60h] [rbp-60h]
  __int64 v96; // [rsp-30h] [rbp-30h]
  __int64 v97; // [rsp-8h] [rbp-8h]

  v3 = *a1;
  v97 = v2;
  v4 = v3;
  v96 = v1;
  switch ( v3 )
  {
    case 0u:
      v31 = *((unsigned int *)a1 + 2);
      if ( (int)v31 >= 0 )
      {
        v32 = a1[3];
        v33 = *(_DWORD *)a1;
        v92 = 0;
        v93 = v31;
        v94 = (v33 >> 8) & 0xFFF;
        v95 = (v32 & 0x10) != 0;
        return sub_CBF760(&v92, 0x20u);
      }
      v72 = *(_QWORD *)(sub_2E88D60(*((_QWORD *)a1 + 2)) + 32);
      v92 = (unsigned __int64)&v94;
      v93 = 0x600000000LL;
      v74 = *((unsigned int *)a1 + 2);
      if ( (int)v74 < 0 )
        v75 = *(_QWORD *)(*(_QWORD *)(v72 + 56) + 16 * (v74 & 0x7FFFFFFF) + 8);
      else
        v75 = *(_QWORD *)(*(_QWORD *)(v72 + 304) + 8 * v74);
      if ( v75 )
      {
        if ( (*(_BYTE *)(v75 + 3) & 0x10) != 0
          || (v75 = *(_QWORD *)(v75 + 32)) != 0 && (*(_BYTE *)(v75 + 3) & 0x10) != 0 )
        {
          v76 = *(_QWORD *)(v75 + 16);
          v77 = 6;
          v78 = 0;
          while ( 1 )
          {
            v79 = *(unsigned __int16 *)(v76 + 68);
            if ( v78 + 1 > (unsigned __int64)v77 )
            {
              sub_C8D5F0((__int64)&v92, &v94, v78 + 1, 8u, v78 + 1, v73);
              v78 = (unsigned int)v93;
            }
            *(_QWORD *)(v92 + 8 * v78) = v79;
            v78 = (unsigned int)(v93 + 1);
            LODWORD(v93) = v93 + 1;
            v80 = *(_QWORD *)(v75 + 16);
            do
            {
              v75 = *(_QWORD *)(v75 + 32);
              if ( !v75 || (*(_BYTE *)(v75 + 3) & 0x10) == 0 )
              {
                v81 = (unsigned __int64 *)v92;
                v82 = 8 * v78;
                goto LABEL_81;
              }
              v76 = *(_QWORD *)(v75 + 16);
            }
            while ( v80 == v76 );
            v77 = HIDWORD(v93);
          }
        }
      }
      v82 = 0;
      v81 = &v94;
LABEL_81:
      result = sub_CBF760(v81, v82);
      if ( (unsigned __int64 *)v92 != &v94 )
      {
        v88 = result;
        _libc_free(v92);
        return v88;
      }
      return result;
    case 1u:
      v34 = *(_DWORD *)a1;
      v15 = *((_QWORD *)a1 + 3);
      v92 = 1;
      v16 = (v34 >> 8) & 0xFFF;
      goto LABEL_16;
    case 2u:
    case 3u:
      v8 = *((_QWORD *)a1 + 3);
      if ( v3 == 2 )
      {
        v83 = *(_DWORD *)(v8 + 32);
        LODWORD(v90) = v83;
        if ( v83 <= 0x40 )
        {
          v10 = &v89;
          v11 = (unsigned __int64)(v83 + 63) >> 6;
          v89 = *(_QWORD *)(v8 + 24);
          goto LABEL_10;
        }
        sub_C43780((__int64)&v89, (const void **)(v8 + 24));
      }
      else
      {
        v9 = (__int64 *)(v8 + 24);
        if ( *(void **)(v8 + 24) == sub_C33340() )
          sub_C3E660((__int64)&v89, (__int64)v9);
        else
          sub_C3A850((__int64)&v89, v9);
      }
      v10 = &v89;
      v11 = ((unsigned __int64)(unsigned int)v90 + 63) >> 6;
      if ( (unsigned int)v90 > 0x40 )
        v10 = (unsigned __int64 *)v89;
LABEL_10:
      v12 = sub_CBF760(v10, 8 * v11);
      v13 = 0;
      if ( *a1 )
        v13 = (*(_DWORD *)a1 >> 8) & 0xFFF;
      v92 = *a1;
      v93 = v13;
      v94 = v12;
      result = sub_CBF760(&v92, 0x18u);
      if ( (unsigned int)v90 > 0x40 && v89 )
      {
        v85 = result;
        j_j___libc_free_0_0(v89);
        return v85;
      }
      return result;
    case 4u:
    case 6u:
    case 0xBu:
    case 0xEu:
      return 0;
    case 5u:
    case 8u:
      v14 = *(_DWORD *)a1;
      v15 = *((int *)a1 + 6);
      v92 = v4;
      v16 = (v14 >> 8) & 0xFFF;
LABEL_16:
      v93 = v16;
      v94 = v15;
      return sub_CBF760(&v92, 0x18u);
    case 7u:
      v6 = (const char *)sub_2EAB930((__int64)a1);
      if ( !v6 )
        return 0;
      v66 = *((unsigned int *)a1 + 2) | (unsigned __int64)((__int64)*((int *)a1 + 8) << 32);
      v67 = strlen(v6);
      v68 = sub_3145F20((__int64)v6, v67);
      v69 = 0;
      if ( *a1 )
        v69 = (*(_DWORD *)a1 >> 8) & 0xFFF;
      v92 = *a1;
      v93 = v69;
      v94 = v68;
      v95 = v66;
      return sub_CBF760(&v92, 0x20u);
    case 9u:
      v35 = *((_QWORD *)a1 + 3);
      v36 = 0;
      if ( v35 )
        v36 = strlen(*((const char **)a1 + 3));
      v37 = sub_3145F20(v35, v36);
      v38 = 0;
      v39 = v37;
      v40 = *((unsigned int *)a1 + 2) | (unsigned __int64)((__int64)*((int *)a1 + 8) << 32);
      v41 = *a1;
      if ( (_BYTE)v41 )
        v38 = (*(_DWORD *)a1 >> 8) & 0xFFF;
      v95 = v39;
      v92 = v41;
      v93 = v38;
      v94 = v40;
      return sub_CBF760(&v92, 0x20u);
    case 0xAu:
      v43 = (_BYTE *)*((_QWORD *)a1 + 3);
      if ( *v43 == 3 )
      {
        v44 = sub_31467A0(*((_QWORD *)a1 + 3), v3);
        if ( v44 )
          goto LABEL_40;
      }
      if ( (v43[7] & 0x10) == 0 )
        return 0;
      v70 = sub_BD5D20((__int64)v43);
      v44 = sub_3145F20((__int64)v70, v71);
LABEL_40:
      v45 = 0;
      v46 = *((unsigned int *)a1 + 2) | (unsigned __int64)((__int64)*((int *)a1 + 8) << 32);
      v47 = *a1;
      if ( (_BYTE)v47 )
        v45 = (*(_DWORD *)a1 >> 8) & 0xFFF;
      v93 = v45;
      v92 = v47;
      v94 = v44;
      v95 = v46;
      return sub_CBF760(&v92, 0x20u);
    case 0xCu:
    case 0xDu:
      v17 = *((_QWORD *)a1 + 2);
      if ( v17 && (v18 = *(_QWORD *)(v17 + 24)) != 0 && (v19 = *(_QWORD *)(v18 + 32)) != 0 )
      {
        v20 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v19 + 16) + 200LL))(*(_QWORD *)(v19 + 16));
        v21 = *((_QWORD *)a1 + 3);
        v22 = (unsigned int)(*(_DWORD *)(v20 + 16) + 31) >> 5;
        v23 = 8 * v22;
        v24 = 4 * v22;
        if ( v24 )
        {
          v25 = (_QWORD *)sub_22077B0(v23);
          for ( i = 0; i != v24; i += 4LL )
            v25[i / 4] = *(unsigned int *)(v21 + i);
        }
        else
        {
          v25 = 0;
          v23 = 0;
        }
        v27 = sub_CBF760(v25, v23);
        v28 = 0;
        if ( *a1 )
          v28 = (*(_DWORD *)a1 >> 8) & 0xFFF;
        v92 = *a1;
        v93 = v28;
        v94 = v27;
        result = sub_CBF760(&v92, 0x18u);
        if ( v25 )
        {
          v86 = result;
          j_j___libc_free_0((unsigned __int64)v25);
          return v86;
        }
      }
      else
      {
        v84 = *(_DWORD *)a1;
        v92 = v3;
        v93 = (v84 >> 8) & 0xFFF;
        return sub_CBF760(&v92, 0x10u);
      }
      return result;
    case 0xFu:
      v48 = *((_QWORD *)a1 + 3);
      if ( (*(_BYTE *)(v48 + 8) & 1) != 0 )
      {
        v49 = *(unsigned __int64 **)(v48 - 8);
        v50 = *v49;
        v51 = (__int64)(v49 + 3);
      }
      else
      {
        v50 = 0;
        v51 = 0;
      }
      v52 = sub_3145F20(v51, v50);
      v53 = 0;
      if ( *a1 )
        v53 = (*(_DWORD *)a1 >> 8) & 0xFFF;
      v92 = *a1;
      v93 = v53;
      v94 = v52;
      return sub_CBF760(&v92, 0x18u);
    case 0x10u:
      v42 = *(_DWORD *)a1;
      v92 = 16;
      v30 = (v42 >> 8) & 0xFFF;
      goto LABEL_28;
    case 0x11u:
      v29 = *(_DWORD *)a1;
      v92 = 17;
      v30 = (v29 >> 8) & 0xFFF;
      goto LABEL_28;
    case 0x12u:
      v54 = *(_DWORD *)a1;
      v92 = 18;
      v30 = (v54 >> 8) & 0xFFF;
LABEL_28:
      v93 = v30;
      v94 = *((unsigned int *)a1 + 6);
      return sub_CBF760(&v92, 0x18u);
    case 0x13u:
      v55 = (int *)*((_QWORD *)a1 + 3);
      v56 = *((_QWORD *)a1 + 4);
      v89 = 0;
      v90 = 0;
      v57 = &v55[v56];
      v91 = 0;
      if ( v55 == v57 )
      {
        v63 = 0;
        v62 = 0;
      }
      else
      {
        v58 = 0;
        v59 = 0;
        while ( 1 )
        {
          v60 = *v55;
          v92 = v60;
          if ( v58 == v59 )
          {
            sub_A235E0((__int64)&v89, v59, &v92);
            v59 = v90;
          }
          else
          {
            if ( v59 )
            {
              *(_QWORD *)v59 = v60;
              v59 = v90;
            }
            v59 += 8;
            v90 = v59;
          }
          if ( v57 == ++v55 )
            break;
          v58 = v91;
        }
        v62 = (_QWORD *)v89;
        v63 = (unsigned __int64)&v59[-v89];
      }
      v64 = sub_CBF760(v62, v63);
      v65 = 0;
      if ( *a1 )
        v65 = (*(_DWORD *)a1 >> 8) & 0xFFF;
      v92 = *a1;
      v93 = v65;
      v94 = v64;
      result = sub_CBF760(&v92, 0x18u);
      if ( v89 )
      {
        v87 = result;
        j_j___libc_free_0(v89);
        return v87;
      }
      return result;
    case 0x14u:
      v61 = *((unsigned int *)a1 + 6);
      v92 = 20;
      v93 = v61;
      v94 = *((unsigned int *)a1 + 7);
      return sub_CBF760(&v92, 0x18u);
    default:
      BUG();
  }
}
