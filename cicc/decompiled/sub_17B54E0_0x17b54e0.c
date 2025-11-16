// Function: sub_17B54E0
// Address: 0x17b54e0
//
_QWORD *__fastcall sub_17B54E0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        double a7,
        double a8,
        double a9)
{
  __int64 v12; // rbx
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r12
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 *v19; // rbx
  __int64 v20; // rax
  __int64 *v21; // rbx
  __int64 v22; // rax
  __int64 *v23; // rbx
  _QWORD *v24; // r15
  __int64 v25; // rax
  __int64 *v26; // rax
  __int64 v27; // r12
  __int64 *v28; // rax
  __int64 v29; // r14
  _QWORD *v30; // r12
  unsigned int v31; // ebx
  __int64 *v32; // r14
  __int64 v33; // r14
  __int64 v34; // rax
  bool v35; // bl
  __int64 v36; // rax
  _QWORD *v37; // rax
  __int64 v39; // rsi
  __int64 v40; // rcx
  __int64 v41; // rax
  __int64 v42; // rdi
  unsigned __int64 *v43; // rbx
  __int64 v44; // rax
  unsigned __int64 v45; // rcx
  __int64 v46; // rsi
  __int64 v47; // rsi
  __int64 v48; // rdx
  unsigned __int8 *v49; // rsi
  unsigned int v50; // ecx
  unsigned int v51; // ebx
  __int64 v52; // rdi
  __int64 v53; // rax
  __int64 v54; // [rsp+8h] [rbp-138h]
  __int64 v55; // [rsp+18h] [rbp-128h]
  __int64 v56; // [rsp+20h] [rbp-120h]
  __int64 v57; // [rsp+20h] [rbp-120h]
  __int64 v58; // [rsp+30h] [rbp-110h]
  __int64 v59; // [rsp+38h] [rbp-108h]
  __int64 v62; // [rsp+50h] [rbp-F0h] BYREF
  unsigned int v63; // [rsp+58h] [rbp-E8h]
  unsigned __int8 *v64; // [rsp+60h] [rbp-E0h] BYREF
  unsigned int v65; // [rsp+68h] [rbp-D8h]
  __int64 v66; // [rsp+70h] [rbp-D0h] BYREF
  unsigned int v67; // [rsp+78h] [rbp-C8h]
  __int16 v68; // [rsp+80h] [rbp-C0h]
  __int64 v69; // [rsp+90h] [rbp-B0h] BYREF
  unsigned int v70; // [rsp+98h] [rbp-A8h]
  __int64 v71; // [rsp+A0h] [rbp-A0h] BYREF
  unsigned int v72; // [rsp+A8h] [rbp-98h]
  __int64 v73; // [rsp+B0h] [rbp-90h] BYREF
  unsigned int v74; // [rsp+B8h] [rbp-88h]
  __int64 v75; // [rsp+C0h] [rbp-80h] BYREF
  unsigned int v76; // [rsp+C8h] [rbp-78h]
  __int64 v77; // [rsp+D0h] [rbp-70h] BYREF
  unsigned int v78; // [rsp+D8h] [rbp-68h]
  __int64 v79; // [rsp+E0h] [rbp-60h] BYREF
  unsigned int v80; // [rsp+E8h] [rbp-58h]
  __int64 v81; // [rsp+F0h] [rbp-50h] BYREF
  unsigned int v82; // [rsp+F8h] [rbp-48h]
  __int64 v83; // [rsp+100h] [rbp-40h]
  unsigned int v84; // [rsp+108h] [rbp-38h]

  v12 = 1;
  while ( 2 )
  {
    switch ( *(_BYTE *)(a2 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v41 = *(_QWORD *)(a2 + 32);
        a2 = *(_QWORD *)(a2 + 24);
        v12 *= v41;
        continue;
      case 1:
        v13 = 16;
        goto LABEL_4;
      case 2:
        v13 = 32;
        goto LABEL_4;
      case 3:
      case 9:
        v13 = 64;
        goto LABEL_4;
      case 4:
        v13 = 80;
        goto LABEL_4;
      case 5:
      case 6:
        v13 = 128;
        goto LABEL_4;
      case 7:
        v13 = 8 * (unsigned int)sub_15A9520(a3, 0);
        goto LABEL_4;
      case 0xB:
        v13 = *(_DWORD *)(a2 + 8) >> 8;
        goto LABEL_4;
      case 0xD:
        v13 = 8LL * *(_QWORD *)sub_15A9930(a3, a2);
        goto LABEL_4;
      case 0xE:
        v58 = *(_QWORD *)(a2 + 24);
        sub_15A9FE0(a3, v58);
        v39 = v58;
        v40 = 1;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v39 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v53 = *(_QWORD *)(v39 + 32);
              v39 = *(_QWORD *)(v39 + 24);
              v40 *= v53;
              continue;
            case 1:
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 9:
            case 0xB:
              goto LABEL_106;
            case 7:
              sub_15A9520(a3, 0);
              goto LABEL_106;
            case 0xD:
              sub_15A9930(a3, v39);
              goto LABEL_106;
            case 0xE:
              v57 = *(_QWORD *)(v39 + 24);
              sub_15A9FE0(a3, v57);
              sub_127FA20(a3, v57);
              goto LABEL_106;
            case 0xF:
              sub_15A9520(a3, *(_DWORD *)(v39 + 8) >> 8);
LABEL_106:
              JUMPOUT(0x17B5DA8);
          }
        }
      case 0xF:
        v13 = 8 * (unsigned int)sub_15A9520(a3, *(_DWORD *)(a2 + 8) >> 8);
LABEL_4:
        v14 = sub_14109A0(a4, a1, a3, a4, (__int64)a5, a6);
        v59 = v15;
        v16 = v14;
        if ( !v14 || !v15 )
          return 0;
        v17 = 0;
        if ( *(_BYTE *)(v14 + 16) == 13 )
          v17 = v14;
        v55 = v17;
        v54 = sub_15A9650(a3, *a1);
        v56 = sub_15A0680(v54, (unsigned __int64)(v12 * v13 + 7) >> 3, 0);
        v18 = sub_146F1B0(a6, v16);
        v19 = sub_1477920(a6, v18, 0);
        v70 = *((_DWORD *)v19 + 2);
        if ( v70 > 0x40 )
        {
          sub_16A4FD0((__int64)&v69, (const void **)v19);
          v72 = *((_DWORD *)v19 + 6);
          if ( v72 <= 0x40 )
            goto LABEL_10;
        }
        else
        {
          v69 = *v19;
          v72 = *((_DWORD *)v19 + 6);
          if ( v72 <= 0x40 )
          {
LABEL_10:
            v71 = v19[2];
            goto LABEL_11;
          }
        }
        sub_16A4FD0((__int64)&v71, (const void **)v19 + 2);
LABEL_11:
        v20 = sub_146F1B0(a6, v59);
        v21 = sub_1477920(a6, v20, 0);
        v74 = *((_DWORD *)v21 + 2);
        if ( v74 > 0x40 )
          sub_16A4FD0((__int64)&v73, (const void **)v21);
        else
          v73 = *v21;
        v76 = *((_DWORD *)v21 + 6);
        if ( v76 > 0x40 )
          sub_16A4FD0((__int64)&v75, (const void **)v21 + 2);
        else
          v75 = v21[2];
        v22 = sub_146F1B0(a6, v56);
        v23 = sub_1477920(a6, v22, 0);
        v78 = *((_DWORD *)v23 + 2);
        if ( v78 > 0x40 )
          sub_16A4FD0((__int64)&v77, (const void **)v23);
        else
          v77 = *v23;
        v80 = *((_DWORD *)v23 + 6);
        if ( v80 > 0x40 )
          sub_16A4FD0((__int64)&v79, (const void **)v23 + 2);
        else
          v79 = v23[2];
        v68 = 257;
        if ( *(_BYTE *)(v16 + 16) > 0x10u || *(_BYTE *)(v59 + 16) > 0x10u )
        {
          LOWORD(v83) = 257;
          v24 = (_QWORD *)sub_15FB440(13, (__int64 *)v16, v59, (__int64)&v81, 0);
          v42 = a5[1];
          if ( v42 )
          {
            v43 = (unsigned __int64 *)a5[2];
            sub_157E9D0(v42 + 40, (__int64)v24);
            v44 = v24[3];
            v45 = *v43;
            v24[4] = v43;
            v45 &= 0xFFFFFFFFFFFFFFF8LL;
            v24[3] = v45 | v44 & 7;
            *(_QWORD *)(v45 + 8) = v24 + 3;
            *v43 = *v43 & 7 | (unsigned __int64)(v24 + 3);
          }
          sub_164B780((__int64)v24, &v66);
          v46 = *a5;
          if ( *a5 )
          {
            v64 = (unsigned __int8 *)*a5;
            sub_1623A60((__int64)&v64, v46, 2);
            v47 = v24[6];
            v48 = (__int64)(v24 + 6);
            if ( v47 )
            {
              sub_161E7C0((__int64)(v24 + 6), v47);
              v48 = (__int64)(v24 + 6);
            }
            v49 = v64;
            v24[6] = v64;
            if ( v49 )
              sub_1623210((__int64)&v64, v49, v48);
          }
        }
        else
        {
          v24 = (_QWORD *)sub_15A2B60((__int64 *)v16, v59, 0, 0, a7, a8, a9);
          v25 = sub_14DBA30((__int64)v24, a5[8], 0);
          if ( v25 )
            v24 = (_QWORD *)v25;
        }
        sub_158AAD0((__int64)&v64, (__int64)&v69);
        sub_158A9F0((__int64)&v66, (__int64)&v73);
        if ( (int)sub_16A9900((__int64)&v64, (unsigned __int64 *)&v66) < 0 )
        {
          LOWORD(v83) = 257;
          v27 = (__int64)sub_17B5310(a5, 36, v16, v59, &v81);
        }
        else
        {
          v26 = (__int64 *)sub_16498A0((__int64)a1);
          v27 = sub_159C540(v26);
        }
        if ( v67 > 0x40 && v66 )
          j_j___libc_free_0_0(v66);
        if ( v65 > 0x40 && v64 )
          j_j___libc_free_0_0(v64);
        sub_158E4C0((__int64)&v81, (__int64)&v69, (__int64)&v73);
        sub_158AAD0((__int64)&v62, (__int64)&v81);
        sub_158A9F0((__int64)&v64, (__int64)&v77);
        if ( (int)sub_16A9900((__int64)&v62, (unsigned __int64 *)&v64) < 0 )
        {
          v68 = 257;
          v29 = (__int64)sub_17B5310(a5, 36, (__int64)v24, v56, &v66);
        }
        else
        {
          v28 = (__int64 *)sub_16498A0((__int64)a1);
          v29 = sub_159C540(v28);
        }
        if ( v65 > 0x40 && v64 )
          j_j___libc_free_0_0(v64);
        if ( v63 > 0x40 && v62 )
          j_j___libc_free_0_0(v62);
        if ( v84 > 0x40 && v83 )
          j_j___libc_free_0_0(v83);
        if ( v82 > 0x40 && v81 )
          j_j___libc_free_0_0(v81);
        LOWORD(v83) = 257;
        v30 = sub_17B51C0(a5, v27, v29, &v81, a7, a8, a9);
        if ( !v55 )
          goto LABEL_49;
        v31 = *(_DWORD *)(v55 + 32);
        v32 = *(__int64 **)(v55 + 24);
        if ( v31 > 0x40 )
        {
          v50 = v31 - 1;
          v51 = v31 + 1;
          v52 = v55 + 24;
          if ( (v32[v50 >> 6] & (1LL << v50)) != 0 )
          {
            if ( v51 - (unsigned int)sub_16A5810(v52) > 0x40 )
              goto LABEL_49;
          }
          else if ( v51 - (unsigned int)sub_16A57B0(v52) > 0x40 )
          {
            goto LABEL_53;
          }
          v33 = *v32;
        }
        else
        {
          v33 = (__int64)((_QWORD)v32 << (64 - (unsigned __int8)v31)) >> (64 - (unsigned __int8)v31);
        }
        if ( v33 >= 0 )
          goto LABEL_53;
LABEL_49:
        sub_158ACE0((__int64)&v81, (__int64)&v69);
        v34 = 1LL << ((unsigned __int8)v82 - 1);
        if ( v82 > 0x40 )
        {
          v35 = (*(_QWORD *)(v81 + 8LL * ((v82 - 1) >> 6)) & v34) != 0;
          if ( v81 )
            j_j___libc_free_0_0(v81);
        }
        else
        {
          v35 = (v81 & v34) != 0;
        }
        if ( v35 )
        {
          LOWORD(v83) = 257;
          v36 = sub_15A0680(v54, 0, 0);
          v37 = sub_17B5310(a5, 40, v59, v36, &v81);
          LOWORD(v83) = 257;
          v30 = sub_17B51C0(a5, (__int64)v37, (__int64)v30, &v81, a7, a8, a9);
        }
LABEL_53:
        if ( v80 > 0x40 && v79 )
          j_j___libc_free_0_0(v79);
        if ( v78 > 0x40 && v77 )
          j_j___libc_free_0_0(v77);
        if ( v76 > 0x40 && v75 )
          j_j___libc_free_0_0(v75);
        if ( v74 > 0x40 && v73 )
          j_j___libc_free_0_0(v73);
        if ( v72 > 0x40 && v71 )
          j_j___libc_free_0_0(v71);
        if ( v70 > 0x40 && v69 )
          j_j___libc_free_0_0(v69);
        return v30;
    }
  }
}
