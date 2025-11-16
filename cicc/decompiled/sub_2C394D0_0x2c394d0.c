// Function: sub_2C394D0
// Address: 0x2c394d0
//
__int64 __fastcall sub_2C394D0(__int64 a1)
{
  _QWORD *v1; // r12
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rcx
  __int64 v31; // rdx
  __m128i *v32; // rsi
  __int64 v33; // r14
  __int64 v34; // rax
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // r13
  __int64 v38; // rbx
  __int64 v39; // rax
  char v40; // di
  char v41; // al
  _QWORD *v42; // rdi
  int v43; // r15d
  __int64 v44; // rax
  __int64 *v45; // rax
  __int64 v46; // rbx
  char v47; // al
  __int64 v48; // r12
  __int64 v49; // r14
  __int64 v50; // rax
  __int64 v51; // r12
  __int64 *v52; // rdi
  __int64 v53; // rsi
  _BYTE *v54; // r13
  _QWORD *v55; // r14
  __int64 v56; // rax
  __int64 v57; // r9
  __int64 v58; // r13
  _QWORD *v59; // rdi
  unsigned __int64 *v60; // rax
  __int64 *v61; // r12
  __int64 *i; // rbx
  __int64 v63; // rax
  __int64 v64; // r12
  __int64 *v65; // rbx
  __int64 *v66; // r14
  __int64 v67; // rax
  _QWORD *v69; // rdi
  __int64 v70; // r12
  __int64 v71; // rsi
  _BYTE *v72; // r13
  _QWORD *v73; // rdi
  __int64 v74; // rsi
  _BYTE *v75; // r13
  _QWORD *v76; // rdi
  __int64 v77; // rsi
  _BYTE *v78; // r13
  __int64 v79; // rax
  _QWORD *v80; // [rsp+8h] [rbp-6F8h]
  unsigned __int8 v81; // [rsp+30h] [rbp-6D0h]
  bool v82; // [rsp+38h] [rbp-6C8h]
  __int64 v83; // [rsp+40h] [rbp-6C0h]
  unsigned __int8 *v84; // [rsp+40h] [rbp-6C0h]
  __int64 v86; // [rsp+50h] [rbp-6B0h]
  __int64 v87; // [rsp+58h] [rbp-6A8h]
  _QWORD *v88; // [rsp+58h] [rbp-6A8h]
  __m128i v89; // [rsp+60h] [rbp-6A0h] BYREF
  __int64 v90; // [rsp+70h] [rbp-690h] BYREF
  __int64 v91; // [rsp+78h] [rbp-688h]
  __int64 v92; // [rsp+80h] [rbp-680h]
  __int64 v93; // [rsp+88h] [rbp-678h]
  _QWORD *v94; // [rsp+90h] [rbp-670h]
  __int64 v95; // [rsp+98h] [rbp-668h]
  _QWORD v96[15]; // [rsp+A0h] [rbp-660h] BYREF
  char v97[120]; // [rsp+118h] [rbp-5E8h] BYREF
  _QWORD v98[12]; // [rsp+190h] [rbp-570h] BYREF
  __int64 v99; // [rsp+1F0h] [rbp-510h]
  __int64 v100; // [rsp+1F8h] [rbp-508h]
  __int16 v101; // [rsp+208h] [rbp-4F8h]
  _QWORD v102[12]; // [rsp+210h] [rbp-4F0h] BYREF
  __m128i *v103; // [rsp+270h] [rbp-490h]
  __int64 v104; // [rsp+278h] [rbp-488h]
  __int16 v105; // [rsp+288h] [rbp-478h]
  __int16 v106; // [rsp+298h] [rbp-468h]
  _QWORD v107[12]; // [rsp+2A0h] [rbp-460h] BYREF
  __m128i *v108; // [rsp+300h] [rbp-400h]
  __int64 v109; // [rsp+308h] [rbp-3F8h]
  __int16 v110; // [rsp+318h] [rbp-3E8h]
  _QWORD v111[15]; // [rsp+320h] [rbp-3E0h] BYREF
  __int16 v112; // [rsp+398h] [rbp-368h]
  __int16 v113[64]; // [rsp+3A8h] [rbp-358h] BYREF
  char v114[136]; // [rsp+428h] [rbp-2D8h] BYREF
  __m128i v115; // [rsp+4B0h] [rbp-250h] BYREF
  __int64 v116; // [rsp+4C0h] [rbp-240h]
  __int16 v117; // [rsp+528h] [rbp-1D8h]
  _BYTE v118[120]; // [rsp+530h] [rbp-1D0h] BYREF
  __int16 v119; // [rsp+5A8h] [rbp-158h]
  __int16 v120; // [rsp+5B8h] [rbp-148h]
  _BYTE v121[120]; // [rsp+5C0h] [rbp-140h] BYREF
  __int16 v122; // [rsp+638h] [rbp-C8h]
  _BYTE v123[120]; // [rsp+640h] [rbp-C0h] BYREF
  __int16 v124; // [rsp+6B8h] [rbp-48h]
  __int16 v125; // [rsp+6C8h] [rbp-38h]

  v1 = v98;
  sub_2C2F4B0(v96, *(_QWORD *)a1);
  v94 = v96;
  v90 = 0;
  v91 = 0;
  v92 = 0;
  v93 = 0;
  v95 = 0;
  sub_2C2B410((__int64)v98, (__int64)v96, v2, v3, v4, v5);
  sub_2C31AD0((__int64)v107, (__int64)v98, v6, v7, v8, v9);
  sub_2C2B5D0((__int64)&v115, v107, v10, v11, v12, v13);
  sub_2AB1B50((__int64)v114);
  sub_2AB1B50((__int64)v113);
  sub_2AB1B50((__int64)v111);
  sub_2AB1B50((__int64)v107);
  sub_2AB1B50((__int64)v102);
  sub_2AB1B50((__int64)v98);
  sub_2ABCC20(v98, (__int64)&v115, v14, v15, v16, v17);
  v101 = v117;
  sub_2ABCC20(v102, (__int64)v118, v18, v19, v20, v21);
  v105 = v119;
  v106 = v120;
  sub_2ABCC20(v107, (__int64)v121, v22, v23, v24, v25);
  v110 = v122;
  sub_2ABCC20(v111, (__int64)v123, v26, v27, v28, v29);
  v30 = v99;
  v31 = v100;
  v112 = v124;
  v113[0] = v125;
LABEL_2:
  v32 = v108;
  if ( v31 - v30 != v109 - (_QWORD)v108 )
    goto LABEL_3;
  if ( v31 != v30 )
  {
    while ( *(_QWORD *)v30 == v32->m128i_i64[0] )
    {
      v41 = *(_BYTE *)(v30 + 24);
      if ( v41 != v32[1].m128i_i8[8]
        || v41 && (*(_QWORD *)(v30 + 8) != v32->m128i_i64[1] || *(_QWORD *)(v30 + 16) != v32[1].m128i_i64[0]) )
      {
        break;
      }
      v30 += 32;
      v32 += 2;
      if ( v30 == v31 )
        goto LABEL_28;
    }
LABEL_3:
    v33 = *(_QWORD *)(v31 - 32);
    v34 = sub_2BF04D0(v33);
    if ( *(_BYTE *)(v33 + 128) )
    {
      if ( *(_DWORD *)(v34 + 88) == 2 )
      {
        v37 = **(_QWORD **)(v34 + 80);
        if ( (unsigned int)*(unsigned __int8 *)(v37 + 8) - 1 <= 1 )
        {
          v38 = 0;
          if ( *(_DWORD *)(v37 + 88) == 1 )
            v38 = **(_QWORD **)(v37 + 80);
          if ( v38 == sub_2BF0520(v33) && v37 + 112 != *(_QWORD *)(v37 + 120) )
          {
            v80 = v1;
            v64 = *(_QWORD *)(v37 + 120);
            do
            {
              if ( !v64 )
                BUG();
              v31 = *(_QWORD *)(v64 + 24);
              v65 = (__int64 *)(v31 + 8LL * *(unsigned int *)(v64 + 32));
              v66 = (__int64 *)v31;
              if ( (__int64 *)v31 != v65 )
              {
                do
                {
                  v67 = sub_2BF0490(*v66);
                  if ( v67 )
                  {
                    v31 = *(unsigned __int8 *)(v67 + 8);
                    switch ( *(_BYTE *)(v67 + 8) )
                    {
                      case 0:
                      case 3:
                      case 5:
                      case 0x13:
                      case 0x14:
                      case 0x15:
                      case 0x16:
                      case 0x1A:
                        break;
                      case 1:
                      case 2:
                      case 4:
                      case 6:
                      case 7:
                      case 8:
                      case 9:
                      case 0xA:
                      case 0xB:
                      case 0xC:
                      case 0xD:
                      case 0xE:
                      case 0xF:
                      case 0x10:
                      case 0x11:
                      case 0x12:
                      case 0x17:
                      case 0x18:
                      case 0x19:
                      case 0x1B:
                      case 0x1C:
                      case 0x1D:
                      case 0x1E:
                      case 0x1F:
                      case 0x20:
                      case 0x21:
                      case 0x22:
                      case 0x23:
                      case 0x24:
                        v32 = &v89;
                        v89.m128i_i64[0] = v37;
                        v89.m128i_i64[1] = v67;
                        sub_2C393B0((__int64)&v90, &v89);
                        break;
                      default:
                        goto LABEL_115;
                    }
                  }
                  ++v66;
                }
                while ( v65 != v66 );
              }
              v64 = *(_QWORD *)(v64 + 8);
            }
            while ( v37 + 112 != v64 );
            v1 = v80;
          }
        }
      }
    }
    while ( 1 )
    {
      sub_2AD7320((__int64)v1, (__int64)v32, v31, v30, v35, v36);
      v31 = v100;
      v30 = v99;
      v32 = v103;
      if ( v100 - v99 == v104 - (_QWORD)v103 )
      {
        if ( v99 == v100 )
          goto LABEL_2;
        v39 = v99;
        while ( *(_QWORD *)v39 == v32->m128i_i64[0] )
        {
          v40 = *(_BYTE *)(v39 + 24);
          if ( v40 != v32[1].m128i_i8[8]
            || v40 && (*(_QWORD *)(v39 + 8) != v32->m128i_i64[1] || *(_QWORD *)(v39 + 16) != v32[1].m128i_i64[0]) )
          {
            break;
          }
          v39 += 32;
          v32 += 2;
          if ( v100 == v39 )
            goto LABEL_2;
        }
      }
      if ( !*(_BYTE *)(*(_QWORD *)(v100 - 32) + 8LL) )
        goto LABEL_2;
    }
  }
LABEL_28:
  sub_2AB1B50((__int64)v111);
  sub_2AB1B50((__int64)v107);
  sub_2AB1B50((__int64)v102);
  sub_2AB1B50((__int64)v1);
  sub_2AB1B50((__int64)v123);
  sub_2AB1B50((__int64)v121);
  sub_2AB1B50((__int64)v118);
  sub_2AB1B50((__int64)&v115);
  if ( *(_DWORD *)(a1 + 88) == 1 )
  {
    v82 = 0;
    v79 = *(_QWORD *)(a1 + 80);
    if ( !*(_BYTE *)(v79 + 4) )
      v82 = *(_DWORD *)v79 == 1;
  }
  else
  {
    v82 = 0;
  }
  v42 = v94;
  if ( (_DWORD)v95 )
  {
    v81 = 0;
    v43 = 0;
    v44 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v45 = &v42[2 * v44];
        v46 = v45[1];
        v86 = *v45;
        if ( *(_QWORD *)(v46 + 80) == *v45 )
          goto LABEL_33;
        if ( sub_2C1AB20(v46) || (unsigned __int8)sub_2C1ABE0(v46) || sub_2C1AA80(v46) )
          goto LABEL_32;
        v47 = *(_BYTE *)(v46 + 8);
        if ( v47 == 9 )
        {
          if ( !v82 && *(_BYTE *)(v46 + 160) )
            goto LABEL_32;
        }
        else if ( v47 != 11 )
        {
          goto LABEL_32;
        }
        v89.m128i_i8[0] = 0;
        v48 = *(unsigned int *)(v46 + 120);
        v49 = *(_QWORD *)(v46 + 112);
        v116 = v46;
        v48 *= 8;
        v83 = v49 + v48;
        v115.m128i_i64[0] = v86;
        v115.m128i_i64[1] = (__int64)&v89;
        v50 = v48 >> 3;
        v51 = v48 >> 5;
        if ( !v51 )
          break;
        v87 = v49 + 32 * v51;
        while ( 1 )
        {
          v52 = *(__int64 **)v49;
          if ( !*(_QWORD *)v49 )
            BUG();
          if ( v52[5] != v115.m128i_i64[0] )
          {
            v53 = v116;
            v54 = (_BYTE *)v115.m128i_i64[1];
            if ( v116 )
              v53 = v116 + 96;
            *v54 = (*(__int64 (__fastcall **)(__int64 *, __int64))(*v52 + 24))(v52, v53);
            if ( !*(_BYTE *)v115.m128i_i64[1] || *(_BYTE *)(v116 + 8) != 9 )
              break;
          }
          v69 = *(_QWORD **)(v49 + 8);
          v70 = v49 + 8;
          if ( !v69 )
            BUG();
          if ( v69[5] != v115.m128i_i64[0] )
          {
            v71 = v116;
            v72 = (_BYTE *)v115.m128i_i64[1];
            if ( v116 )
              v71 = v116 + 96;
            *v72 = (*(__int64 (__fastcall **)(_QWORD *, __int64))(*v69 + 24LL))(v69, v71);
            if ( !*(_BYTE *)v115.m128i_i64[1] || *(_BYTE *)(v116 + 8) != 9 )
              goto LABEL_80;
          }
          v73 = *(_QWORD **)(v49 + 16);
          v70 = v49 + 16;
          if ( !v73 )
            BUG();
          if ( v73[5] != v115.m128i_i64[0] )
          {
            v74 = v116;
            v75 = (_BYTE *)v115.m128i_i64[1];
            if ( v116 )
              v74 = v116 + 96;
            *v75 = (*(__int64 (__fastcall **)(_QWORD *, __int64))(*v73 + 24LL))(v73, v74);
            if ( !*(_BYTE *)v115.m128i_i64[1] || *(_BYTE *)(v116 + 8) != 9 )
              goto LABEL_80;
          }
          v76 = *(_QWORD **)(v49 + 24);
          v70 = v49 + 24;
          if ( !v76 )
            BUG();
          if ( v76[5] != v115.m128i_i64[0] )
          {
            v77 = v116;
            v78 = (_BYTE *)v115.m128i_i64[1];
            if ( v116 )
              v77 = v116 + 96;
            *v78 = (*(__int64 (__fastcall **)(_QWORD *, __int64))(*v76 + 24LL))(v76, v77);
            if ( !*(_BYTE *)v115.m128i_i64[1] || *(_BYTE *)(v116 + 8) != 9 )
            {
LABEL_80:
              v49 = v70;
              break;
            }
          }
          v49 += 32;
          if ( v87 == v49 )
          {
            v50 = (v83 - v49) >> 3;
            goto LABEL_95;
          }
        }
LABEL_48:
        if ( v83 == v49 )
          goto LABEL_49;
LABEL_32:
        v42 = v94;
LABEL_33:
        v44 = (unsigned int)(v43 + 1);
        v43 = v44;
        if ( (_DWORD)v44 == (_DWORD)v95 )
          goto LABEL_70;
      }
LABEL_95:
      if ( v50 != 2 )
      {
        if ( v50 != 3 )
        {
          if ( v50 != 1 )
            goto LABEL_49;
          goto LABEL_98;
        }
        if ( !sub_2C25030(&v115, *(__int64 **)v49) )
          goto LABEL_48;
        v49 += 8;
      }
      if ( !sub_2C25030(&v115, *(__int64 **)v49) )
        goto LABEL_48;
      v49 += 8;
LABEL_98:
      if ( !sub_2C25030(&v115, *(__int64 **)v49) )
        goto LABEL_48;
LABEL_49:
      if ( v89.m128i_i8[0] )
      {
        if ( !v82 )
        {
          v88 = *(_QWORD **)(v46 + 48);
          v55 = &v88[*(unsigned int *)(v46 + 56)];
          v84 = *(unsigned __int8 **)(v46 + 136);
          v56 = sub_22077B0(0xA8u);
          v58 = v56;
          if ( v56 )
          {
            sub_2ABDBC0(v56, 9, v88, v55, v84, v57);
            v59 = (_QWORD *)v58;
            v58 += 96;
            *(_QWORD *)(v58 - 96) = &unk_4A237B0;
            *(_QWORD *)v58 = &unk_4A23830;
            *(_WORD *)(v58 + 64) = 1;
            *(_QWORD *)(v58 - 56) = &unk_4A237F8;
            sub_2C19D60(v59, v46);
          }
          else
          {
            sub_2C19D60(0, v46);
          }
          v115.m128i_i64[0] = v86;
          sub_2BF1090(
            v46 + 96,
            v58,
            (unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD))sub_2C250A0,
            (__int64)&v115);
          goto LABEL_54;
        }
        goto LABEL_32;
      }
LABEL_54:
      v60 = (unsigned __int64 *)sub_2BF05A0(v86);
      sub_2C19EE0((_QWORD *)v46, v86, v60);
      v61 = *(__int64 **)(v46 + 48);
      for ( i = &v61[*(unsigned int *)(v46 + 56)]; i != v61; ++v61 )
      {
        v63 = sub_2BF0490(*v61);
        if ( v63 )
        {
          switch ( *(_BYTE *)(v63 + 8) )
          {
            case 0:
            case 3:
            case 5:
            case 0x13:
            case 0x14:
            case 0x15:
            case 0x16:
            case 0x1A:
              continue;
            case 1:
            case 2:
            case 4:
            case 6:
            case 7:
            case 8:
            case 9:
            case 0xA:
            case 0xB:
            case 0xC:
            case 0xD:
            case 0xE:
            case 0xF:
            case 0x10:
            case 0x11:
            case 0x12:
            case 0x17:
            case 0x18:
            case 0x19:
            case 0x1B:
            case 0x1C:
            case 0x1D:
            case 0x1E:
            case 0x1F:
            case 0x20:
            case 0x21:
            case 0x22:
            case 0x23:
            case 0x24:
              v115.m128i_i64[0] = v86;
              v115.m128i_i64[1] = v63;
              sub_2C393B0((__int64)&v90, &v115);
              break;
            default:
LABEL_115:
              BUG();
          }
        }
      }
      v44 = (unsigned int)(v43 + 1);
      v81 = 1;
      v42 = v94;
      v43 = v44;
      if ( (_DWORD)v44 == (_DWORD)v95 )
        goto LABEL_70;
    }
  }
  v81 = 0;
LABEL_70:
  if ( v42 != v96 )
    _libc_free((unsigned __int64)v42);
  sub_C7D6A0(v91, 16LL * (unsigned int)v93, 8);
  sub_2AB1B50((__int64)v97);
  sub_2AB1B50((__int64)v96);
  return v81;
}
