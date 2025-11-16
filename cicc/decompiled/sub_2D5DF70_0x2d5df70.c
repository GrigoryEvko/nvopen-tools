// Function: sub_2D5DF70
// Address: 0x2d5df70
//
__int64 __fastcall sub_2D5DF70(__int64 a1, __int64 a2)
{
  unsigned __int64 *v2; // rdx
  __int64 v3; // rcx
  unsigned __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // r15
  __int64 v12; // rsi
  __int64 v13; // rbx
  __int64 v14; // r13
  __int64 *v15; // rbx
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdi
  unsigned int v19; // eax
  __int64 v20; // r9
  unsigned int v21; // ecx
  __int64 v22; // rax
  __int64 v23; // r11
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // r10
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rsi
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // r15
  unsigned __int64 *v34; // rax
  int v35; // ecx
  unsigned __int64 *v36; // rdx
  __int64 (__fastcall *v37)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v38; // rdi
  __int64 (*v39)(); // rax
  _QWORD *v40; // rax
  __int64 v41; // rdx
  char *v42; // rbx
  char *v43; // r13
  __int64 v44; // rdx
  unsigned int v45; // esi
  unsigned __int64 v46; // rsi
  unsigned __int64 v47; // r13
  __int64 v49; // [rsp+0h] [rbp-1B0h]
  __int64 v50; // [rsp+8h] [rbp-1A8h]
  __int64 v52; // [rsp+20h] [rbp-190h]
  __int64 v53; // [rsp+28h] [rbp-188h]
  unsigned __int64 v54; // [rsp+30h] [rbp-180h]
  __int64 v56; // [rsp+40h] [rbp-170h]
  __int64 v57; // [rsp+48h] [rbp-168h]
  unsigned __int8 v58; // [rsp+50h] [rbp-160h]
  char v59; // [rsp+60h] [rbp-150h]
  __int64 v60; // [rsp+68h] [rbp-148h]
  __int64 v61; // [rsp+70h] [rbp-140h]
  char v62; // [rsp+7Fh] [rbp-131h]
  __int64 v63; // [rsp+80h] [rbp-130h]
  int v64; // [rsp+88h] [rbp-128h]
  unsigned __int64 v65; // [rsp+88h] [rbp-128h]
  char v66[32]; // [rsp+90h] [rbp-120h] BYREF
  __int16 v67; // [rsp+B0h] [rbp-100h]
  _QWORD v68[4]; // [rsp+C0h] [rbp-F0h] BYREF
  __int16 v69; // [rsp+E0h] [rbp-D0h]
  char *v70; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 v71; // [rsp+F8h] [rbp-B8h]
  _BYTE v72[32]; // [rsp+100h] [rbp-B0h] BYREF
  __int64 v73; // [rsp+120h] [rbp-90h]
  __int64 v74; // [rsp+128h] [rbp-88h]
  __int64 v75; // [rsp+130h] [rbp-80h]
  __int64 v76; // [rsp+138h] [rbp-78h]
  void **v77; // [rsp+140h] [rbp-70h]
  void **v78; // [rsp+148h] [rbp-68h]
  __int64 v79; // [rsp+150h] [rbp-60h]
  int v80; // [rsp+158h] [rbp-58h]
  __int16 v81; // [rsp+15Ch] [rbp-54h]
  char v82; // [rsp+15Eh] [rbp-52h]
  __int64 v83; // [rsp+160h] [rbp-50h]
  __int64 v84; // [rsp+168h] [rbp-48h]
  void *v85; // [rsp+170h] [rbp-40h] BYREF
  void *v86; // [rsp+178h] [rbp-38h] BYREF

  v2 = *(unsigned __int64 **)(a2 - 8);
  v54 = *v2;
  if ( *(_BYTE *)*v2 != 17 )
  {
    v60 = *(_QWORD *)(a2 + 40);
    v64 = *(_DWORD *)(a2 + 4);
    v57 = *(_QWORD *)(*v2 + 8);
    v50 = ((v64 & 0x7FFFFFFu) >> 1) - 1;
    if ( (v64 & 0x7FFFFFFu) >> 1 != 1 )
    {
      v52 = 0;
      v58 = 0;
      while ( 1 )
      {
        v3 = (unsigned int)(2 * ++v52);
        v4 = v2[4 * v3];
        v5 = 4;
        if ( (_DWORD)v52 != -1 )
          v5 = 4LL * (unsigned int)(v3 + 1);
        v53 = v2[v5];
        v6 = sub_AA5930(v53);
        v56 = v7;
        if ( v6 != v7 )
          break;
LABEL_51:
        if ( v52 == v50 )
          return v58;
        v2 = *(unsigned __int64 **)(a2 - 8);
      }
      v59 = 0;
      v8 = v6;
      v65 = v4;
      while ( 1 )
      {
        v61 = *(_QWORD *)(v8 + 8);
        if ( *(_BYTE *)(v61 + 8) != 12
          || *(_DWORD *)(v57 + 8) >> 8 >= *(_DWORD *)(v61 + 8) >> 8
          || (v38 = *(_QWORD *)(a1 + 16), v39 = *(__int64 (**)())(*(_QWORD *)v38 + 1424LL), v39 == sub_2D56670)
          || (v62 = ((__int64 (__fastcall *)(__int64, __int64, __int64))v39)(v38, v57, v61)) == 0 )
        {
          if ( v57 != v61 )
            goto LABEL_8;
          v62 = 0;
        }
        if ( (*(_DWORD *)(v8 + 4) & 0x7FFFFFF) != 0 )
        {
          v10 = 0;
          v11 = 0;
          v63 = 8LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF);
          while ( 1 )
          {
            v12 = *(_QWORD *)(v8 - 8);
            v13 = 4 * v10;
            v14 = *(_QWORD *)(v12 + 4 * v10);
            if ( v65 != v14 )
            {
              if ( !v62 || *(_BYTE *)v14 != 17 )
                goto LABEL_24;
              sub_C449B0((__int64)&v70, (const void **)(v65 + 24), *(_DWORD *)(v61 + 8) >> 8);
              if ( *(_DWORD *)(v14 + 32) <= 0x40u )
              {
                if ( *(char **)(v14 + 24) != v70 )
                {
LABEL_21:
                  if ( (unsigned int)v71 > 0x40 && v70 )
                    j_j___libc_free_0_0((unsigned __int64)v70);
                  goto LABEL_24;
                }
              }
              else if ( !sub_C43C50(v14 + 24, (const void **)&v70) )
              {
                goto LABEL_21;
              }
              if ( (unsigned int)v71 > 0x40 && v70 )
                j_j___libc_free_0_0((unsigned __int64)v70);
              v12 = *(_QWORD *)(v8 - 8);
            }
            if ( v60 == *(_QWORD *)(v12 + 32LL * *(unsigned int *)(v8 + 72) + v10) )
              break;
LABEL_24:
            v10 += 8;
            if ( v63 == v10 )
              goto LABEL_8;
          }
          if ( !v59 )
          {
            v18 = *(_QWORD *)(a2 - 8);
            if ( v53 == *(_QWORD *)(v18 + 32) )
              goto LABEL_51;
            v19 = (*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1;
            v20 = v19 - 1;
            if ( v19 == 1 )
              goto LABEL_51;
            v21 = 2;
            v22 = 1;
            v23 = 0;
            do
            {
              while ( 1 )
              {
                v25 = 32;
                if ( (_DWORD)v22 != -1 )
                  v25 = 32LL * (v21 + 1);
                if ( v53 == *(_QWORD *)(v18 + v25) )
                  break;
                v24 = v22;
                v21 += 2;
                ++v22;
                if ( v20 == v24 )
                  goto LABEL_50;
              }
              if ( v23 )
                goto LABEL_51;
              v26 = v21;
              v27 = v22;
              v21 += 2;
              ++v22;
              v23 = *(_QWORD *)(v18 + 32 * v26);
            }
            while ( v20 != v27 );
LABEL_50:
            if ( !v23 )
              goto LABEL_51;
          }
          if ( v11 )
          {
LABEL_29:
            v15 = (__int64 *)(v12 + v13);
            if ( !*v15 || (v16 = v15[1], (*(_QWORD *)v15[2] = v16) == 0) )
            {
              *v15 = v11;
LABEL_35:
              v17 = *(_QWORD *)(v11 + 16);
              v15[1] = v17;
              if ( v17 )
                *(_QWORD *)(v17 + 16) = v15 + 1;
              v15[2] = v11 + 16;
              v59 = 1;
              *(_QWORD *)(v11 + 16) = v15;
              v58 = 1;
              goto LABEL_24;
            }
LABEL_31:
            *(_QWORD *)(v16 + 16) = v15[2];
LABEL_32:
            *v15 = v11;
            if ( v11 )
              goto LABEL_35;
            v59 = 1;
            v58 = 1;
            goto LABEL_24;
          }
          if ( v65 == v14 )
          {
            v11 = v54;
            goto LABEL_29;
          }
          v28 = sub_BD5C60(a2);
          v82 = 7;
          v76 = v28;
          v77 = &v85;
          v78 = &v86;
          v70 = v72;
          v85 = &unk_49DA100;
          v71 = 0x200000000LL;
          v79 = 0;
          v86 = &unk_49DA0B0;
          v29 = *(_QWORD *)(a2 + 40);
          v80 = 0;
          v73 = v29;
          v81 = 512;
          v83 = 0;
          v84 = 0;
          v74 = a2 + 24;
          LOWORD(v75) = 0;
          v30 = *(_QWORD *)sub_B46C60(a2);
          v68[0] = v30;
          if ( !v30 || (sub_B96E90((__int64)v68, v30, 1), (v33 = v68[0]) == 0) )
          {
            sub_93FB40((__int64)&v70, 0);
            v33 = v68[0];
            goto LABEL_79;
          }
          v34 = (unsigned __int64 *)v70;
          v35 = v71;
          v36 = (unsigned __int64 *)&v70[16 * (unsigned int)v71];
          if ( v70 != (char *)v36 )
          {
            while ( *(_DWORD *)v34 )
            {
              v34 += 2;
              if ( v36 == v34 )
                goto LABEL_81;
            }
            v34[1] = v68[0];
            goto LABEL_61;
          }
LABEL_81:
          if ( (unsigned int)v71 >= (unsigned __int64)HIDWORD(v71) )
          {
            v46 = (unsigned int)v71 + 1LL;
            v47 = v49 & 0xFFFFFFFF00000000LL;
            v49 &= 0xFFFFFFFF00000000LL;
            if ( HIDWORD(v71) < v46 )
            {
              sub_C8D5F0((__int64)&v70, v72, v46, 0x10u, v31, v32);
              v36 = (unsigned __int64 *)&v70[16 * (unsigned int)v71];
            }
            *v36 = v47;
            v36[1] = v33;
            v33 = v68[0];
            LODWORD(v71) = v71 + 1;
          }
          else
          {
            if ( v36 )
            {
              *(_DWORD *)v36 = 0;
              v36[1] = v33;
              v35 = v71;
              v33 = v68[0];
            }
            LODWORD(v71) = v35 + 1;
          }
LABEL_79:
          if ( v33 )
LABEL_61:
            sub_B91220((__int64)v68, v33);
          v67 = 257;
          if ( v61 == *(_QWORD *)(v54 + 8) )
          {
            v11 = v54;
          }
          else
          {
            v37 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, __int64))*((_QWORD *)*v77 + 15);
            if ( v37 != sub_920130 )
            {
              v11 = v37((__int64)v77, 39u, (_BYTE *)v54, v61);
              goto LABEL_67;
            }
            if ( *(_BYTE *)v54 > 0x15u )
              goto LABEL_87;
            v11 = (unsigned __int8)sub_AC4810(0x27u)
                ? sub_ADAB70(39, v54, (__int64 **)v61, 0)
                : sub_AA93C0(0x27u, v54, v61);
LABEL_67:
            if ( !v11 )
            {
LABEL_87:
              v69 = 257;
              v40 = sub_BD2C40(72, 1u);
              v11 = (__int64)v40;
              if ( v40 )
                sub_B515B0((__int64)v40, v54, v61, (__int64)v68, 0, 0);
              (*((void (__fastcall **)(void **, __int64, char *, __int64, __int64))*v78 + 2))(v78, v11, v66, v74, v75);
              v41 = 16LL * (unsigned int)v71;
              if ( v70 != &v70[v41] )
              {
                v42 = v70;
                v43 = &v70[v41];
                do
                {
                  v44 = *((_QWORD *)v42 + 1);
                  v45 = *(_DWORD *)v42;
                  v42 += 16;
                  sub_B99FD0(v11, v45, v44);
                }
                while ( v43 != v42 );
                v13 = 4 * v10;
              }
            }
          }
          nullsub_61();
          v85 = &unk_49DA100;
          nullsub_63();
          if ( v70 != v72 )
            _libc_free((unsigned __int64)v70);
          v15 = (__int64 *)(*(_QWORD *)(v8 - 8) + v13);
          if ( *v15 )
          {
            v16 = v15[1];
            *(_QWORD *)v15[2] = v16;
            if ( v16 )
              goto LABEL_31;
          }
          goto LABEL_32;
        }
LABEL_8:
        v9 = *(_QWORD *)(v8 + 32);
        if ( !v9 )
          BUG();
        v8 = 0;
        if ( *(_BYTE *)(v9 - 24) == 84 )
          v8 = v9 - 24;
        if ( v56 == v8 )
          goto LABEL_51;
      }
    }
  }
  return 0;
}
