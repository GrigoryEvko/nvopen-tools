// Function: sub_2AC5DB0
// Address: 0x2ac5db0
//
__int64 __fastcall sub_2AC5DB0(__int64 *a1, unsigned __int8 *a2, __int64 a3, __int64 a4)
{
  void *v4; // r8
  char v8; // al
  __int64 v9; // r9
  signed __int64 v10; // r13
  __int64 v11; // r9
  __int64 *v12; // r15
  int v13; // edx
  __int64 v14; // rax
  __int64 v15; // r13
  __int64 *v16; // r13
  __int64 v17; // r9
  __int64 v18; // r12
  unsigned __int8 *v19; // r8
  _QWORD *v20; // rdx
  __int64 *v21; // rcx
  signed __int64 v23; // r13
  __int64 v24; // r9
  _QWORD *v25; // rax
  __int64 v26; // rax
  __int64 v27; // r12
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  unsigned __int64 v34; // r13
  __int64 *v35; // r15
  __int64 v36; // r15
  __int64 *v37; // rax
  __int64 *v38; // rdi
  __int64 *v39; // rdi
  signed __int64 v40; // r13
  __int64 v41; // r15
  __int64 v42; // rsi
  __int64 v43; // rax
  __int64 v44; // r15
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 *v47; // r12
  __int64 v48; // r15
  __int64 v49; // rax
  __int64 v50; // r12
  __int64 v51; // r15
  __int64 *v52; // rdi
  __int64 v53; // r13
  __int64 *v54; // rax
  void *srca; // [rsp+0h] [rbp-C0h]
  void *v58; // [rsp+8h] [rbp-B8h]
  int v59; // [rsp+8h] [rbp-B8h]
  int v60; // [rsp+8h] [rbp-B8h]
  void *v61; // [rsp+8h] [rbp-B8h]
  __int64 v62; // [rsp+10h] [rbp-B0h] BYREF
  int v63; // [rsp+18h] [rbp-A8h]
  void *v64; // [rsp+20h] [rbp-A0h] BYREF
  __int16 v65; // [rsp+40h] [rbp-80h]
  __int64 *v66; // [rsp+50h] [rbp-70h] BYREF
  __int64 v67; // [rsp+58h] [rbp-68h]
  _BYTE dest[96]; // [rsp+60h] [rbp-60h] BYREF

  v4 = (void *)a3;
  switch ( *a2 )
  {
    case ')':
    case '*':
    case '+':
    case ',':
    case '-':
    case '.':
    case '/':
    case '2':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
    case ':':
    case ';':
    case 'R':
    case 'S':
    case 'V':
    case '`':
      goto LABEL_3;
    case '0':
    case '1':
    case '3':
    case '4':
      v8 = sub_2AB37C0(a1[5], a2);
      v4 = (void *)a3;
      if ( !v8 )
      {
LABEL_3:
        v10 = 8 * a4;
        v66 = (__int64 *)dest;
        v67 = 0x600000000LL;
        v11 = v10 >> 3;
        if ( (unsigned __int64)v10 > 0x30 )
        {
          srca = v4;
          sub_C8D5F0((__int64)&v66, dest, v10 >> 3, 8u, (__int64)v4, v11);
          v11 = v10 >> 3;
          v4 = srca;
          v38 = &v66[(unsigned int)v67];
        }
        else
        {
          v12 = (__int64 *)dest;
          if ( !v10 )
            goto LABEL_5;
          v38 = (__int64 *)dest;
        }
        v59 = v11;
        memcpy(v38, v4, v10);
        v12 = v66;
        LODWORD(v10) = v67;
        LODWORD(v11) = v59;
LABEL_5:
        v13 = *a2;
        LODWORD(v67) = v11 + v10;
        v14 = (unsigned int)(v11 + v10);
        if ( (unsigned int)(v13 - 42) <= 0x11 )
        {
          v58 = *(void **)(a1[6] + 112);
          if ( (_BYTE)v13 == 46 )
          {
            v51 = *v12;
            if ( !sub_2BF04A0(v51) )
            {
              v53 = *(_QWORD *)(v51 + 40);
              if ( *(_BYTE *)v53 > 0x15u && sub_D97040((__int64)v58, *(_QWORD *)(v53 + 8)) )
              {
                v54 = sub_DD8400((__int64)v58, v53);
                if ( !*((_WORD *)v54 + 12) )
                  v51 = sub_2AC42A0(*a1, v54[4]);
              }
            }
            *v66 = v51;
            v12 = v66;
          }
          v15 = v12[1];
          if ( !sub_2BF04A0(v15) )
          {
            v36 = *(_QWORD *)(v15 + 40);
            if ( *(_BYTE *)v36 > 0x15u && sub_D97040((__int64)v58, *(_QWORD *)(v36 + 8)) )
            {
              v37 = sub_DD8400((__int64)v58, v36);
              if ( !*((_WORD *)v37 + 12) )
                v15 = sub_2AC42A0(*a1, v37[4]);
            }
          }
          v12 = v66;
          v66[1] = v15;
          v14 = (unsigned int)v67;
        }
        v16 = &v12[v14];
        v18 = sub_22077B0(0xA8u);
        if ( v18 )
        {
          v19 = a2;
          v20 = v12;
          v21 = v16;
          goto LABEL_11;
        }
        goto LABEL_12;
      }
      v40 = 8 * a4;
      v66 = (__int64 *)dest;
      v67 = 0x600000000LL;
      v41 = v40 >> 3;
      if ( (unsigned __int64)v40 > 0x30 )
      {
        sub_C8D5F0((__int64)&v66, dest, v40 >> 3, 8u, a3, v9);
        v4 = (void *)a3;
        v52 = &v66[(unsigned int)v67];
      }
      else
      {
        if ( !v40 )
          goto LABEL_30;
        v52 = (__int64 *)dest;
      }
      memcpy(v52, v4, v40);
      LODWORD(v40) = v67;
LABEL_30:
      v42 = *((_QWORD *)a2 + 5);
      LODWORD(v67) = v41 + v40;
      v43 = sub_2AB6F10((__int64)a1, v42);
      v44 = *a1;
      v61 = (void *)v43;
      v45 = sub_AD64C0(*((_QWORD *)a2 + 1), 1, 0);
      v46 = sub_2AC42A0(v44, v45);
      v47 = (__int64 *)a1[7];
      v48 = v46;
      v49 = *((_QWORD *)a2 + 6);
      v65 = 257;
      v62 = v49;
      if ( v49 )
        sub_2AAAFA0(&v62);
      v50 = sub_2AAF680(v47, (__int64)v61, v66[1], v48, &v62, &v64, v63, 0);
      sub_9C6650(&v62);
      v34 = (unsigned __int64)v66;
      v66[1] = v50;
      v35 = (__int64 *)(v34 + 8LL * (unsigned int)v67);
      v18 = sub_22077B0(0xA8u);
      if ( v18 )
        goto LABEL_19;
LABEL_12:
      if ( v66 != (__int64 *)dest )
        _libc_free((unsigned __int64)v66);
      return v18;
    case ']':
      v23 = 8 * a4;
      v66 = (__int64 *)dest;
      v67 = 0x600000000LL;
      v24 = (8 * a4) >> 3;
      if ( (unsigned __int64)(8 * a4) > 0x30 )
      {
        sub_C8D5F0((__int64)&v66, dest, v23 >> 3, 8u, a3, v24);
        v24 = v23 >> 3;
        v4 = (void *)a3;
        v39 = &v66[(unsigned int)v67];
      }
      else
      {
        if ( !v23 )
          goto LABEL_18;
        v39 = (__int64 *)dest;
      }
      v60 = v24;
      memcpy(v39, v4, v23);
      LODWORD(v23) = v67;
      LODWORD(v24) = v60;
LABEL_18:
      LODWORD(v67) = v24 + v23;
      v25 = (_QWORD *)sub_BD5C60((__int64)a2);
      v26 = sub_BCB2D0(v25);
      v27 = *a1;
      v28 = sub_AD64C0(v26, **((unsigned int **)a2 + 9), 0);
      v29 = sub_2AC42A0(v27, v28);
      sub_2AB9420((__int64)&v66, v29, v30, v31, v32, v33);
      v34 = (unsigned __int64)v66;
      v35 = &v66[(unsigned int)v67];
      v18 = sub_22077B0(0xA8u);
      if ( v18 )
      {
LABEL_19:
        v19 = a2;
        v20 = (_QWORD *)v34;
        v21 = v35;
LABEL_11:
        sub_2ABDBC0(v18, 23, v20, v21, v19, v17);
        *(_QWORD *)v18 = &unk_4A23EC8;
        *(_QWORD *)(v18 + 40) = &unk_4A23F00;
        *(_QWORD *)(v18 + 96) = &unk_4A23F38;
        *(_DWORD *)(v18 + 160) = *a2 - 29;
      }
      goto LABEL_12;
    default:
      return 0;
  }
}
