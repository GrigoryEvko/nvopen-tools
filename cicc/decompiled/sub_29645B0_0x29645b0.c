// Function: sub_29645B0
// Address: 0x29645b0
//
__int64 __fastcall sub_29645B0(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  _BYTE *v8; // rdi
  __int64 v9; // r14
  __int64 v10; // rdx
  __int64 v11; // rcx
  unsigned __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // r15
  __int64 v16; // rbx
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 *v21; // rsi
  __int64 *v22; // rax
  _QWORD *v23; // rax
  unsigned __int64 v24; // r14
  unsigned __int8 **v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  _QWORD *v30; // rax
  _QWORD *v31; // r14
  __int64 v32; // rsi
  __int64 v33; // rdx
  __int64 *v34; // rax
  unsigned __int64 v35; // rdx
  __int64 *v36; // r12
  _QWORD *v37; // r15
  _QWORD *v38; // r14
  __int64 v39; // r13
  __int64 v40; // r15
  __int64 v41; // rax
  __int64 v42; // rsi
  _QWORD *v43; // rax
  _QWORD *v44; // rdx
  unsigned __int8 v45; // bl
  _QWORD *v47; // rax
  _QWORD *v48; // rax
  _QWORD *v49; // rax
  __int64 v50; // rsi
  unsigned __int8 *v51; // rsi
  _QWORD *v52; // rax
  _QWORD *v53; // rax
  _QWORD *v54; // rax
  __int64 v55; // rsi
  _QWORD *v56; // r14
  __int64 v57; // rdx
  _QWORD *v58; // rax
  __int64 v59; // rsi
  unsigned __int8 *v60; // rsi
  __int64 v61; // [rsp+0h] [rbp-E0h]
  __int64 v62; // [rsp+8h] [rbp-D8h]
  int v63; // [rsp+10h] [rbp-D0h]
  char v64; // [rsp+10h] [rbp-D0h]
  _QWORD *v65; // [rsp+10h] [rbp-D0h]
  __int64 v66; // [rsp+18h] [rbp-C8h]
  __int64 v67; // [rsp+18h] [rbp-C8h]
  __int64 v68; // [rsp+20h] [rbp-C0h]
  __int64 v69; // [rsp+28h] [rbp-B8h]
  __int64 v72; // [rsp+40h] [rbp-A0h]
  char v74; // [rsp+55h] [rbp-8Bh]
  unsigned __int8 v75; // [rsp+56h] [rbp-8Ah]
  char v76; // [rsp+57h] [rbp-89h]
  __int64 *v77; // [rsp+58h] [rbp-88h]
  __int64 v79; // [rsp+68h] [rbp-78h]
  __int64 v80; // [rsp+68h] [rbp-78h]
  __int64 v81; // [rsp+78h] [rbp-68h] BYREF
  __int64 v82; // [rsp+80h] [rbp-60h] BYREF
  __int64 v83; // [rsp+88h] [rbp-58h]
  _BYTE v84[16]; // [rsp+90h] [rbp-50h] BYREF
  __int16 v85; // [rsp+A0h] [rbp-40h]

  v8 = (_BYTE *)*(a2 - 12);
  v81 = 0;
  v9 = (__int64)sub_2958930(v8);
  v76 = sub_D48480((__int64)a1, v9, v10, v11);
  if ( v76 )
  {
    v81 = v9 & 0xFFFFFFFFFFFFFFFBLL;
  }
  else
  {
    if ( *(_BYTE *)v9 > 0x1Cu )
    {
      sub_295B990(&v82, (__int64)a1, v9);
      if ( (v82 & 0xFFFFFFFFFFFFFFF8LL) != 0 && ((v82 & 4) == 0 || *(_DWORD *)((v82 & 0xFFFFFFFFFFFFFFF8LL) + 8)) )
      {
        v14 = v82;
        v82 = 0;
        v81 = v14;
      }
      else if ( ((v81 >> 2) & 1) == 0 )
      {
        v81 = 0;
      }
      sub_295C970(&v82);
    }
    v12 = v81 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v81 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_67;
    if ( (v81 & 4) != 0 )
    {
      if ( !*(_DWORD *)(v12 + 8) )
        goto LABEL_67;
    }
    else
    {
      v76 = 0;
    }
  }
  v15 = *(a2 - 4);
  v16 = (__int64)(a1 + 7);
  if ( !(unsigned __int8)sub_B19060((__int64)(a1 + 7), v15, v12, v13) )
  {
    v74 = 1;
    v19 = 0x1FFFFFFFFFFFFFF8LL;
    v63 = 0;
    goto LABEL_14;
  }
  v15 = *(a2 - 8);
  v74 = sub_B19060((__int64)(a1 + 7), v15, v17, v18);
  if ( v74 )
  {
LABEL_67:
    v75 = 0;
    goto LABEL_68;
  }
  v63 = 1;
  v19 = 0x1FFFFFFFFFFFFFFCLL;
LABEL_14:
  v72 = (__int64)&a2[v19];
  v66 = a2[v19];
  v68 = a2[5];
  v75 = sub_2957D10((__int64)a1, v68, v15, v18);
  if ( !v75 )
    goto LABEL_67;
  if ( !v76 && (v74 ? !sub_2957E60((unsigned __int8 *)v9) : !sub_2957DE0(v9)) )
    goto LABEL_67;
  if ( a5 )
  {
    v21 = sub_2958C70(v15, a4);
    if ( v21 )
      sub_DAC210(a5, (unsigned __int64)v21);
    else
      sub_DAC8B0(a5, a1);
    sub_D9D700(a5, 0);
  }
  if ( a6 && byte_4F8F8E8[0] )
    nullsub_390();
  v79 = sub_D4B130((__int64)a1);
  v22 = (__int64 *)a1[4];
  v85 = 257;
  v77 = &v82;
  v62 = sub_F41C30(v79, *v22, a3, a4, a6, (void **)&v82);
  if ( !v76 )
  {
    v85 = 257;
    v69 = sub_F36960(v15, *(__int64 **)(v15 + 56), 1, a3, a4, a6, (void **)&v82, 0);
    if ( !a6 || !byte_4F8F8E8[0] )
    {
      v23 = (_QWORD *)sub_986580(v79);
      sub_B43D60(v23);
LABEL_30:
      v24 = sub_986580(v79);
      v64 = qword_5005C28;
      v25 = (unsigned __int8 **)sub_295C9D0(&v81);
      sub_2959DB0(v79, v25, v26, v74, v69, v62, v64, v24, 0, a3);
      goto LABEL_31;
    }
    goto LABEL_91;
  }
  if ( !sub_AA5510(v15) )
  {
    v85 = 257;
    v69 = sub_F36960(v15, *(__int64 **)(v15 + 56), 1, a3, a4, a6, (void **)&v82, 0);
    if ( a6 )
    {
      if ( !byte_4F8F8E8[0] )
      {
        v52 = (_QWORD *)sub_986580(v79);
        sub_B43D60(v52);
        goto LABEL_77;
      }
LABEL_91:
      nullsub_390();
      v49 = (_QWORD *)sub_986580(v79);
      sub_B43D60(v49);
      if ( !v76 )
        goto LABEL_30;
      goto LABEL_77;
    }
    v58 = (_QWORD *)sub_986580(v79);
    sub_B43D60(v58);
LABEL_103:
    sub_B44550(a2, v79, (unsigned __int64 *)(v79 + 48), 0);
    sub_AC2B30((__int64)(a2 - 12), v9);
    sub_B43C20((__int64)&v82, v68);
    v54 = sub_F340F0(v66, v82, v83);
    v55 = a2[6];
    v56 = v54;
    v82 = v55;
    if ( v55 )
    {
      sub_B96E90((__int64)&v82, v55, 1);
      v57 = (__int64)(v56 + 6);
      if ( v56 + 6 == &v82 )
      {
        if ( v82 )
          sub_B91220((__int64)&v82, v82);
        goto LABEL_78;
      }
      v59 = v56[6];
      if ( !v59 )
      {
LABEL_113:
        v60 = (unsigned __int8 *)v82;
        v56[6] = v82;
        if ( v60 )
          sub_B976B0((__int64)&v82, v60, v57);
        goto LABEL_78;
      }
    }
    else
    {
      v57 = (__int64)(v54 + 6);
      if ( v54 + 6 == &v82 )
        goto LABEL_78;
      v59 = v54[6];
      if ( !v59 )
        goto LABEL_78;
    }
    v61 = v57;
    sub_B91220(v57, v59);
    v57 = v61;
    goto LABEL_113;
  }
  if ( !a6 )
  {
    v53 = (_QWORD *)sub_986580(v79);
    sub_B43D60(v53);
    v69 = v15;
    goto LABEL_103;
  }
  if ( byte_4F8F8E8[0] )
  {
    v69 = v15;
    goto LABEL_91;
  }
  v47 = (_QWORD *)sub_986580(v79);
  sub_B43D60(v47);
  v69 = v15;
LABEL_77:
  sub_B44550(a2, v79, (unsigned __int64 *)(v79 + 48), 0);
  sub_AC2B30((__int64)(a2 - 12), v9);
  v48 = (_QWORD *)sub_B47F80(a2);
  sub_B44240(v48, v68, (unsigned __int64 *)(v68 + 48), 0);
LABEL_78:
  sub_AC2B30((__int64)&a2[-4 * v63 - 4], v69);
  sub_AC2B30(v72, v62);
LABEL_31:
  sub_B24BB0(a3, v79, v69);
  if ( !a6 )
  {
    if ( v76 )
      sub_B20B50(a3, v68, v15);
    goto LABEL_42;
  }
  v83 = 0x100000000LL;
  v82 = (__int64)v84;
  sub_F35FA0((__int64)&v82, v79, v69 & 0xFFFFFFFFFFFFFFFBLL, v27, v28, v29);
  sub_D724E0(a6, v82, (unsigned int)v83, a3);
  if ( (_BYTE *)v82 != v84 )
    _libc_free(v82);
  if ( v76 )
  {
    v65 = (_QWORD *)sub_986580(v68);
    sub_B43C20((__int64)&v82, v68);
    v30 = sub_F340F0(v66, v82, v83);
    v31 = v30;
    v32 = v65[6];
    v82 = v32;
    if ( v32 )
    {
      sub_B96E90((__int64)&v82, v32, 1);
      v33 = (__int64)(v31 + 6);
      if ( v31 + 6 == &v82 )
      {
        if ( v82 )
          sub_B91220((__int64)&v82, v82);
        goto LABEL_39;
      }
      v50 = v31[6];
      if ( !v50 )
      {
LABEL_96:
        v51 = (unsigned __int8 *)v82;
        v31[6] = v82;
        if ( v51 )
          sub_B976B0((__int64)&v82, v51, v33);
        goto LABEL_39;
      }
    }
    else
    {
      v33 = (__int64)(v30 + 6);
      if ( v30 + 6 == &v82 || (v50 = v30[6]) == 0 )
      {
LABEL_39:
        sub_B43D60(v65);
        sub_D6D7F0(a6, v68, v15);
        sub_B20B50(a3, v68, v15);
        goto LABEL_40;
      }
    }
    v67 = v33;
    sub_B91220(v33, v50);
    v33 = v67;
    goto LABEL_96;
  }
LABEL_40:
  if ( byte_4F8F8E8[0] )
    nullsub_390();
LABEL_42:
  if ( v15 == v69 )
    sub_29584C0(v15, v79);
  else
    sub_2959A40(v15, v69, v68, v79, v76);
  v34 = (__int64 *)sub_BD5C60((__int64)a2);
  if ( v74 )
    v80 = sub_ACD720(v34);
  else
    v80 = sub_ACD6D0(v34);
  v35 = v81 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v81 & 4) == 0 )
  {
    if ( !v35 )
      goto LABEL_62;
    v36 = &v81;
    goto LABEL_49;
  }
  v36 = *(__int64 **)v35;
  v77 = (__int64 *)(*(_QWORD *)v35 + 8LL * *(unsigned int *)(v35 + 8));
  if ( *(__int64 **)v35 != v77 )
  {
LABEL_49:
    v37 = a1;
    while ( 1 )
    {
      v38 = v37;
      v39 = *(_QWORD *)(*v36 + 16);
      if ( !v39 )
        goto LABEL_60;
      do
      {
        v40 = v39;
        v39 = *(_QWORD *)(v39 + 8);
        v41 = *(_QWORD *)(v40 + 24);
        if ( *(_BYTE *)v41 <= 0x1Cu )
          continue;
        v42 = *(_QWORD *)(v41 + 40);
        if ( *((_BYTE *)v38 + 84) )
        {
          v43 = (_QWORD *)v38[8];
          v44 = &v43[*((unsigned int *)v38 + 19)];
          if ( v43 != v44 )
          {
            while ( v42 != *v43 )
            {
              if ( v44 == ++v43 )
                goto LABEL_58;
            }
LABEL_57:
            sub_AC2B30(v40, v80);
          }
        }
        else if ( sub_C8CA60(v16, v42) )
        {
          goto LABEL_57;
        }
LABEL_58:
        ;
      }
      while ( v39 );
      v37 = v38;
LABEL_60:
      if ( v77 == ++v36 )
      {
        a1 = v37;
        break;
      }
    }
  }
LABEL_62:
  if ( v76 )
    sub_2963C70((__int64)a1, v62, a3, a4, a6, a5);
  if ( a6 )
  {
    v45 = byte_4F8F8E8[0];
    if ( byte_4F8F8E8[0] )
    {
      nullsub_390();
      v75 = v45;
    }
  }
LABEL_68:
  sub_295C970(&v81);
  return v75;
}
