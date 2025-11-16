// Function: sub_DF7390
// Address: 0xdf7390
//
_BOOL8 __fastcall sub_DF7390(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int8 *v8; // rax
  __int64 v9; // rdi
  unsigned __int8 *v10; // rcx
  bool v11; // cf
  unsigned __int8 *v12; // rax
  __int64 v13; // rdx
  _BOOL8 v14; // r13
  __int64 v16; // r12
  signed __int64 v17; // rbx
  _BYTE *v18; // r14
  unsigned __int64 v19; // r15
  unsigned __int64 v20; // r13
  __int64 v21; // rdx
  unsigned __int64 v22; // r10
  _BYTE *v23; // rax
  _QWORD *v24; // rbx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rdx
  __int64 v30; // rax
  _BYTE *v31; // rax
  __int64 v32; // rdx
  __int64 v33; // r10
  __int64 v34; // rax
  unsigned __int64 v35; // rsi
  __int64 v36; // rdx
  unsigned __int64 v37; // r10
  unsigned int v38; // eax
  __int64 v39; // rbx
  char v40; // al
  __int64 v41; // rax
  _BYTE *v42; // rax
  __int64 v43; // rsi
  unsigned __int64 v44; // r13
  __int64 v45; // rax
  __int64 v46; // rax
  unsigned __int8 *v47; // [rsp+8h] [rbp-A8h]
  unsigned int v48; // [rsp+14h] [rbp-9Ch]
  unsigned __int64 v49; // [rsp+18h] [rbp-98h]
  __int64 v50; // [rsp+28h] [rbp-88h]
  unsigned __int64 v51; // [rsp+28h] [rbp-88h]
  unsigned __int64 v52; // [rsp+30h] [rbp-80h]
  unsigned __int64 v53; // [rsp+30h] [rbp-80h]
  __int64 v54; // [rsp+30h] [rbp-80h]
  char v55; // [rsp+30h] [rbp-80h]
  __int64 v56; // [rsp+30h] [rbp-80h]
  unsigned __int64 v57; // [rsp+30h] [rbp-80h]
  __int64 v58; // [rsp+30h] [rbp-80h]
  __int64 v59; // [rsp+38h] [rbp-78h]
  __int64 v61; // [rsp+48h] [rbp-68h]
  char *v62; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v63; // [rsp+58h] [rbp-58h]
  __int64 v64; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v65; // [rsp+68h] [rbp-48h]
  __int64 v66; // [rsp+70h] [rbp-40h] BYREF
  __int64 v67; // [rsp+78h] [rbp-38h]

  v8 = sub_BD3990((unsigned __int8 *)a3, a2);
  v9 = *a1;
  v10 = v8;
  v11 = *v8 < 4u;
  v12 = 0;
  if ( v11 )
    v12 = v10;
  v47 = v12;
  v48 = sub_AE43A0(v9, *(_QWORD *)(a3 + 8));
  v63 = v48;
  if ( v48 > 0x40 )
    sub_C43690((__int64)&v62, 0, 0);
  else
    v62 = 0;
  if ( a5 )
  {
    v59 = a4 + 8 * a5;
    if ( a4 != v59 )
    {
      v16 = a4 + 8;
      v49 = 0;
      v17 = a2 & 0xFFFFFFFFFFFFFFF9LL | 4;
      while ( 1 )
      {
        v61 = v16;
        v18 = *(_BYTE **)(v16 - 8);
        v19 = v17 & 0xFFFFFFFFFFFFFFF8LL;
        v20 = v17 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v17 )
        {
          v30 = sub_BCBAE0(v19, *(unsigned __int8 **)(v16 - 8), v13);
          v18 = *(_BYTE **)(v16 - 8);
          v22 = v30;
          if ( *v18 != 17 )
          {
            v53 = v30;
            v31 = (_BYTE *)sub_9B7920(*(char **)(v16 - 8));
            v22 = v53;
            v18 = v31;
            if ( v31 )
            {
              if ( *v31 != 17 )
                v18 = 0;
            }
          }
          goto LABEL_42;
        }
        v21 = (v17 >> 1) & 3;
        if ( v21 == 2 )
          break;
        if ( v21 == 1 )
        {
          if ( v19 )
          {
            v22 = *(_QWORD *)(v19 + 24);
            if ( *v18 == 17 )
              goto LABEL_60;
LABEL_19:
            v52 = v22;
            v23 = (_BYTE *)sub_9B7920(*(char **)(v16 - 8));
            v22 = v52;
            v21 = (v17 >> 1) & 3;
            v18 = v23;
            if ( !v23 )
              goto LABEL_23;
            if ( *v23 != 17 )
              v18 = 0;
LABEL_22:
            v21 = (v17 >> 1) & 3;
            goto LABEL_23;
          }
LABEL_69:
          v41 = sub_BCBAE0(0, *(unsigned __int8 **)(v16 - 8), v21);
          v18 = *(_BYTE **)(v16 - 8);
          v21 = (v17 >> 1) & 3;
          v22 = v41;
          if ( *v18 == 17 )
            goto LABEL_60;
LABEL_70:
          v57 = v22;
          v42 = (_BYTE *)sub_9B7920(v18);
          v22 = v57;
          v18 = v42;
          if ( v42 && *v42 != 17 )
            v18 = 0;
          goto LABEL_22;
        }
        v46 = sub_BCBAE0(v17 & 0xFFFFFFFFFFFFFFF8LL, *(unsigned __int8 **)(v16 - 8), v21);
        v18 = *(_BYTE **)(v16 - 8);
        v21 = (v17 >> 1) & 3;
        v22 = v46;
        if ( *v18 != 17 )
          goto LABEL_70;
LABEL_23:
        if ( !v21 )
        {
          if ( v19 )
          {
            v24 = (_QWORD *)*((_QWORD *)v18 + 3);
            if ( *((_DWORD *)v18 + 8) > 0x40u )
              v24 = (_QWORD *)*v24;
            v25 = 16LL * (unsigned int)v24 + sub_AE4AC0(*a1, v19) + 24;
            v26 = *(_QWORD *)v25;
            LOBYTE(v25) = *(_BYTE *)(v25 + 8);
            v66 = v26;
            LOBYTE(v67) = v25;
            v27 = sub_CA1930(&v66);
            sub_C46A40((__int64)&v62, v27);
            goto LABEL_28;
          }
LABEL_42:
          if ( sub_BCEA30(v22) )
            goto LABEL_37;
          v33 = *a1;
LABEL_44:
          v54 = v33;
          v34 = sub_BCBAE0(v17 & 0xFFFFFFFFFFFFFFF8LL, *(unsigned __int8 **)(v16 - 8), v32);
          v33 = v54;
          v35 = v34;
LABEL_45:
          v50 = v33;
          v55 = sub_AE5020(v33, v35);
          v66 = sub_9208B0(v50, v35);
          v67 = v36;
          v28 = 1LL << v55;
          v37 = (((unsigned __int64)(v66 + 7) >> 3) + (1LL << v55) - 1) >> v55 << v55;
          goto LABEL_46;
        }
LABEL_60:
        v56 = v21;
        v40 = sub_BCEA30(v22);
        v32 = v56;
        if ( v40 )
          goto LABEL_37;
        v33 = *a1;
        if ( v56 == 2 )
        {
          v35 = v17 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v19 )
            goto LABEL_44;
          goto LABEL_45;
        }
        if ( v56 != 1 )
          goto LABEL_44;
        if ( v19 )
        {
          v43 = *(_QWORD *)(v19 + 24);
        }
        else
        {
          v58 = *a1;
          v45 = sub_BCBAE0(0, *(unsigned __int8 **)(v16 - 8), 1);
          v33 = v58;
          v43 = v45;
        }
        v66 = sub_9208B0(v33, v43);
        v67 = v28;
        v37 = (unsigned __int64)(v66 + 7) >> 3;
LABEL_46:
        if ( v18 )
        {
          v51 = v37;
          sub_C44B10((__int64)&v64, (char **)v18 + 3, v48);
          sub_C47170((__int64)&v64, v51);
          v38 = v65;
          v65 = 0;
          LODWORD(v67) = v38;
          v66 = v64;
          sub_C45EE0((__int64)&v62, &v66);
          if ( (unsigned int)v67 > 0x40 && v66 )
            j_j___libc_free_0_0(v66);
          if ( v65 > 0x40 && v64 )
            j_j___libc_free_0_0(v64);
        }
        else
        {
          if ( v49 )
            goto LABEL_37;
          v49 = v37;
        }
        if ( v17 )
        {
          v39 = (v17 >> 1) & 3;
          if ( v39 == 2 )
          {
            if ( v19 )
              goto LABEL_29;
          }
          else if ( v39 == 1 && v19 )
          {
            v20 = *(_QWORD *)(v19 + 24);
            goto LABEL_29;
          }
        }
LABEL_28:
        v20 = sub_BCBAE0(v19, *(unsigned __int8 **)(v16 - 8), v28);
LABEL_29:
        v13 = *(unsigned __int8 *)(v20 + 8);
        if ( (_BYTE)v13 == 16 )
        {
          v17 = *(_QWORD *)(v20 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
LABEL_14:
          v16 += 8;
          if ( v61 == v59 )
            goto LABEL_32;
        }
        else
        {
          if ( (unsigned int)(unsigned __int8)v13 - 17 > 1 )
          {
            v44 = v20 & 0xFFFFFFFFFFFFFFF9LL;
            v17 = 0;
            if ( (_BYTE)v13 == 15 )
              v17 = v44;
            goto LABEL_14;
          }
          v16 += 8;
          v17 = v20 & 0xFFFFFFFFFFFFFFF9LL | 2;
          if ( v61 == v59 )
            goto LABEL_32;
        }
      }
      v22 = v17 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v19 )
      {
        if ( *v18 == 17 )
          goto LABEL_60;
        goto LABEL_19;
      }
      goto LABEL_69;
    }
    v49 = 0;
LABEL_32:
    sub_C44B10((__int64)&v66, &v62, 0x40u);
    if ( (unsigned int)v67 > 0x40 )
    {
      if ( *(_QWORD *)v66 | (unsigned __int64)v47 || v49 > 1 )
      {
        j_j___libc_free_0_0(v66);
LABEL_37:
        v14 = 1;
        goto LABEL_7;
      }
      j_j___libc_free_0_0(v66);
    }
    else
    {
      v29 = 0;
      if ( (_DWORD)v67 )
        v29 = v66 << (64 - (unsigned __int8)v67) >> (64 - (unsigned __int8)v67);
      if ( v29 | (unsigned __int64)v47 || v49 > 1 )
        goto LABEL_37;
    }
    v14 = 0;
    goto LABEL_7;
  }
  v14 = v47 != 0;
LABEL_7:
  if ( v63 > 0x40 && v62 )
    j_j___libc_free_0_0(v62);
  return v14;
}
