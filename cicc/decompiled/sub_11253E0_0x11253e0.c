// Function: sub_11253E0
// Address: 0x11253e0
//
unsigned __int8 *__fastcall sub_11253E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  signed int v5; // ebx
  __int64 v6; // rdx
  __int16 v7; // r8
  _BYTE *v8; // rax
  __int64 v9; // rdi
  void *v10; // r15
  signed int v11; // eax
  unsigned __int8 *v12; // r12
  __int16 v14; // r8
  void *v15; // rdx
  __int16 v16; // r8
  void *v17; // rdx
  int v18; // eax
  __int64 v19; // rdi
  __int64 v20; // rax
  void *v21; // r15
  void *v22; // rax
  __int64 v23; // rbx
  __int64 v24; // rax
  __int64 v25; // r8
  unsigned __int64 v26; // rax
  int v27; // eax
  int v28; // ebx
  __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v31; // rax
  void *v32; // r15
  void *v33; // rax
  unsigned __int64 v34; // rax
  int v35; // eax
  int v36; // eax
  __int64 v37; // rax
  __int64 v38; // rax
  bool v39; // zf
  __int16 v40; // ax
  __int64 v41; // rax
  __int64 v42; // rbx
  __int64 v43; // r14
  unsigned __int8 *v44; // rax
  __int16 v45; // ax
  __int16 v46; // ax
  __int16 v47; // ax
  int v48; // eax
  _QWORD *v49; // rdi
  signed int v50; // ebx
  __int64 v51; // [rsp+0h] [rbp-B0h]
  void *v52; // [rsp+8h] [rbp-A8h]
  void *v53; // [rsp+8h] [rbp-A8h]
  __int64 v54; // [rsp+8h] [rbp-A8h]
  __int16 v55; // [rsp+10h] [rbp-A0h]
  __int16 v56; // [rsp+10h] [rbp-A0h]
  __int16 v57; // [rsp+10h] [rbp-A0h]
  int v58; // [rsp+10h] [rbp-A0h]
  __int16 v59; // [rsp+10h] [rbp-A0h]
  bool v60; // [rsp+17h] [rbp-99h]
  char v62; // [rsp+20h] [rbp-90h]
  unsigned int v63; // [rsp+20h] [rbp-90h]
  unsigned int v64; // [rsp+24h] [rbp-8Ch]
  __int64 v65; // [rsp+28h] [rbp-88h]
  char v66; // [rsp+3Fh] [rbp-71h] BYREF
  unsigned __int64 v67; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v68; // [rsp+48h] [rbp-68h]
  bool v69; // [rsp+4Ch] [rbp-64h]
  _QWORD v70[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v71; // [rsp+70h] [rbp-40h]

  if ( *(_BYTE *)a4 == 18 )
  {
    v65 = a4 + 24;
  }
  else
  {
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a4 + 8) + 8LL) - 17 > 1 )
      return 0;
    v8 = sub_AD7630(a4, 0, a3);
    if ( !v8 || *v8 != 18 )
      return 0;
    v65 = (__int64)(v8 + 24);
  }
  v5 = sub_BCB090(*(_QWORD *)(a3 + 8));
  if ( v5 == -1 )
    return 0;
  if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
    v6 = *(_QWORD *)(a3 - 8);
  else
    v6 = a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
  v64 = sub_BCB060(*(_QWORD *)(*(_QWORD *)v6 + 8LL));
  v62 = *(_BYTE *)a3;
  v60 = *(_BYTE *)a3 == 72;
  v7 = *(_WORD *)(a2 + 2) & 0x37;
  if ( v7 == 6 || v7 == 1 )
  {
    v66 = 0;
    v68 = v64;
    if ( v64 > 0x40 )
    {
      v59 = v7;
      sub_C43690((__int64)&v67, 0, 0);
      v7 = v59;
    }
    else
    {
      v67 = 0;
    }
    v55 = v7;
    v69 = v60;
    sub_C41980((void **)v65, (__int64)&v67, 1, &v66);
    if ( !v66 )
    {
      v52 = sub_C33340();
      if ( *(void **)v65 == v52 )
      {
        sub_C3C790(v70, (_QWORD **)v65);
        v15 = v52;
        v14 = v55;
      }
      else
      {
        sub_C33EB0(v70, (__int64 *)v65);
        v14 = v55;
        v15 = v52;
      }
      v53 = v15;
      v56 = v14;
      if ( (void *)v70[0] == v15 )
      {
        sub_C3E740(v70, 1u);
        v17 = v53;
        v16 = v56;
      }
      else
      {
        sub_C3BAB0((__int64)v70, 1);
        v16 = v56;
        v17 = v53;
      }
      v57 = v16;
      if ( *(void **)v65 == v17 )
        v18 = sub_C3E510(v65, (__int64)v70);
      else
        v18 = sub_C37950(v65, (__int64)v70);
      if ( v18 != 1 )
      {
        v19 = *(_QWORD *)(a2 + 8);
        if ( v57 == 1 )
          v20 = sub_AD6450(v19);
        else
          v20 = sub_AD6400(v19);
        v12 = sub_F162A0(a1, a2, v20);
        sub_91D830(v70);
        sub_969240((__int64 *)&v67);
        return v12;
      }
      sub_91D830(v70);
    }
    if ( v68 > 0x40 && v67 )
      j_j___libc_free_0_0(v67);
  }
  if ( (int)v64 <= v5 )
    goto LABEL_9;
  v9 = v65;
  v10 = sub_C33340();
  if ( *(void **)v65 == v10 )
    v9 = *(_QWORD *)(v65 + 8);
  v11 = sub_C3BD20(v9);
  if ( v11 == 0x7FFFFFFF )
  {
    if ( *(void **)v65 == v10 )
      sub_C3C500(v70, (__int64)v10);
    else
      sub_C373C0(v70, *(_QWORD *)v65);
    if ( v10 == (void *)v70[0] )
      sub_C3CF90((__int64)v70, 0);
    else
      sub_C35910((__int64)v70, 0);
    v49 = v70;
    if ( v10 == (void *)v70[0] )
      v49 = (_QWORD *)v70[1];
    v50 = sub_C3BD20((__int64)v49);
    sub_91D830(v70);
    if ( (int)(v64 - (v62 != 72)) > v50 )
      return 0;
  }
  else if ( v5 <= v11 && (int)(v64 - (v62 != 72)) >= v11 )
  {
    return 0;
  }
LABEL_9:
  switch ( *(_WORD *)(a2 + 2) & 0x3F )
  {
    case 1:
    case 9:
      v58 = 32;
      goto LABEL_36;
    case 2:
    case 0xA:
      if ( v62 != 72 )
      {
        v58 = 38;
        goto LABEL_37;
      }
      v58 = 34;
      goto LABEL_70;
    case 3:
    case 0xB:
      if ( v62 != 72 )
      {
        v58 = 39;
        goto LABEL_37;
      }
      v58 = 35;
      goto LABEL_70;
    case 4:
    case 0xC:
      if ( v62 != 72 )
      {
        v58 = 40;
        goto LABEL_37;
      }
      v58 = 36;
      goto LABEL_70;
    case 5:
    case 0xD:
      if ( v62 != 72 )
      {
        v58 = 41;
LABEL_37:
        v21 = *(void **)v65;
        v22 = sub_C33340();
        v23 = (__int64)v22;
        if ( v21 == v22 )
          sub_C3C460(v70, (__int64)v22);
        else
          sub_C37380(v70, (__int64)v21);
        v68 = v64;
        v24 = 1LL << ((unsigned __int8)v64 - 1);
        v63 = v64 - 1;
        v51 = v24;
        v25 = ~v24;
        if ( v64 > 0x40 )
        {
          v54 = ~v24;
          sub_C43690((__int64)&v67, -1, 1);
          v25 = v54;
          if ( v68 > 0x40 )
          {
            *(_QWORD *)(v67 + 8LL * (v63 >> 6)) &= v54;
LABEL_44:
            if ( v70[0] == v23 )
              sub_C400C0(v70, (__int64)&v67, 1u, 1u);
            else
              sub_C36910((__int64)v70, (__int64)&v67, 1, 1);
            if ( v68 > 0x40 && v67 )
              j_j___libc_free_0_0(v67);
            if ( v70[0] == v23 )
              v27 = sub_C3E510((__int64)v70, v65);
            else
              v27 = sub_C37950((__int64)v70, v65);
            if ( !v27 )
            {
              v28 = v58;
              v29 = *(_QWORD *)(a2 + 8);
              if ( (unsigned int)(v58 - 40) <= 1 )
                goto LABEL_85;
              goto LABEL_53;
            }
            sub_91D830(v70);
            if ( *(_QWORD *)v65 == v23 )
              sub_C3C460(v70, v23);
            else
              sub_C37380(v70, *(_QWORD *)v65);
            v68 = v64;
            if ( v64 > 0x40 )
            {
              sub_C43690((__int64)&v67, 0, 0);
              if ( v68 > 0x40 )
              {
                *(_QWORD *)(v67 + 8LL * (v63 >> 6)) |= v51;
LABEL_97:
                if ( v23 == v70[0] )
                  sub_C400C0(v70, (__int64)&v67, 1u, 1u);
                else
                  sub_C36910((__int64)v70, (__int64)&v67, 1, 1);
                if ( v68 > 0x40 && v67 )
                  j_j___libc_free_0_0(v67);
                if ( v23 == v70[0] )
                  v36 = sub_C3E510((__int64)v70, v65);
                else
                  v36 = sub_C37950((__int64)v70, v65);
                if ( v36 == 2 )
                {
                  v28 = v58;
                  v29 = *(_QWORD *)(a2 + 8);
                  if ( (unsigned int)(v58 - 38) <= 1 )
                    goto LABEL_85;
LABEL_53:
                  if ( v28 != 33 )
                  {
LABEL_54:
                    v30 = sub_AD6450(v29);
LABEL_55:
                    v12 = sub_F162A0(a1, a2, v30);
                    sub_91D830(v70);
                    return v12;
                  }
LABEL_85:
                  v30 = sub_AD6400(v29);
                  goto LABEL_55;
                }
                goto LABEL_105;
              }
            }
            else
            {
              v67 = 0;
            }
            v67 |= v51;
            goto LABEL_97;
          }
        }
        else
        {
          v26 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v64;
          if ( !v64 )
            v26 = 0;
          v67 = v26;
        }
        v67 &= v25;
        goto LABEL_44;
      }
      v58 = 37;
LABEL_70:
      v32 = *(void **)v65;
      v33 = sub_C33340();
      v23 = (__int64)v33;
      if ( v32 == v33 )
        sub_C3C460(v70, (__int64)v33);
      else
        sub_C37380(v70, (__int64)v32);
      v68 = v64;
      if ( v64 > 0x40 )
      {
        sub_C43690((__int64)&v67, -1, 1);
      }
      else
      {
        v34 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v64;
        if ( !v64 )
          v34 = 0;
        v67 = v34;
      }
      if ( v70[0] == v23 )
        sub_C400C0(v70, (__int64)&v67, 0, 1u);
      else
        sub_C36910((__int64)v70, (__int64)&v67, 0, 1);
      if ( v68 > 0x40 && v67 )
        j_j___libc_free_0_0(v67);
      if ( v70[0] == v23 )
        v35 = sub_C3E510((__int64)v70, v65);
      else
        v35 = sub_C37950((__int64)v70, v65);
      if ( !v35 )
      {
        v28 = v58;
        v29 = *(_QWORD *)(a2 + 8);
        if ( (unsigned int)(v58 - 36) <= 1 )
          goto LABEL_85;
        goto LABEL_53;
      }
      sub_91D830(v70);
      if ( *(_QWORD *)v65 == v23 )
        sub_C3C460(v70, v23);
      else
        sub_C37380(v70, *(_QWORD *)v65);
      v68 = v64;
      if ( v64 > 0x40 )
        sub_C43690((__int64)&v67, 0, 0);
      else
        v67 = 0;
      if ( v23 == v70[0] )
        sub_C400C0(v70, (__int64)&v67, 0, 1u);
      else
        sub_C36910((__int64)v70, (__int64)&v67, 0, 1);
      sub_969240((__int64 *)&v67);
      if ( v23 == v70[0] )
        v48 = sub_C3E510((__int64)v70, v65);
      else
        v48 = sub_C37950((__int64)v70, v65);
      if ( v48 == 2 )
      {
        v29 = *(_QWORD *)(a2 + 8);
        if ( (unsigned int)(v58 - 33) > 2 )
          goto LABEL_54;
        goto LABEL_85;
      }
LABEL_105:
      sub_91D830(v70);
      v68 = v64;
      if ( v64 > 0x40 )
        sub_C43690((__int64)&v67, 0, 0);
      else
        v67 = 0;
      v69 = v60;
      sub_C41980((void **)v65, (__int64)&v67, 0, &v66);
      v37 = v65;
      if ( *(_QWORD *)v65 == v23 )
        v37 = *(_QWORD *)(v65 + 8);
      if ( (*(_BYTE *)(v37 + 20) & 7) == 3 || v66 )
      {
LABEL_129:
        v41 = sub_986520(a3);
        v42 = *(_QWORD *)v41;
        v43 = sub_AD8D80(*(_QWORD *)(*(_QWORD *)v41 + 8LL), (__int64)&v67);
        v71 = 257;
        v44 = (unsigned __int8 *)sub_BD2C40(72, unk_3F10FD0);
        v12 = v44;
        if ( v44 )
          sub_1113300((__int64)v44, v58, v42, v43, (__int64)v70);
      }
      else
      {
        switch ( v58 )
        {
          case '!':
            goto LABEL_123;
          case '"':
            if ( !sub_9696D0((_QWORD *)v65) )
              goto LABEL_129;
            goto LABEL_123;
          case '#':
            if ( !sub_9696D0((_QWORD *)v65) )
            {
              LOWORD(v58) = 34;
              goto LABEL_129;
            }
LABEL_123:
            v38 = sub_AD6400(*(_QWORD *)(a2 + 8));
LABEL_124:
            v12 = sub_F162A0(a1, a2, v38);
            break;
          case '$':
            if ( sub_9696D0((_QWORD *)v65) )
              goto LABEL_132;
            LOWORD(v58) = 37;
            goto LABEL_129;
          case '%':
            if ( !sub_9696D0((_QWORD *)v65) )
              goto LABEL_129;
            goto LABEL_132;
          case '&':
            v39 = !sub_9696D0((_QWORD *)v65);
            v47 = 39;
            if ( v39 )
              v47 = v58;
            LOWORD(v58) = v47;
            goto LABEL_129;
          case '\'':
            v39 = !sub_9696D0((_QWORD *)v65);
            v46 = 38;
            if ( !v39 )
              v46 = v58;
            LOWORD(v58) = v46;
            goto LABEL_129;
          case '(':
            v39 = !sub_9696D0((_QWORD *)v65);
            v45 = 41;
            if ( !v39 )
              v45 = v58;
            LOWORD(v58) = v45;
            goto LABEL_129;
          case ')':
            v39 = !sub_9696D0((_QWORD *)v65);
            v40 = 40;
            if ( v39 )
              v40 = v58;
            LOWORD(v58) = v40;
            goto LABEL_129;
          default:
LABEL_132:
            v38 = sub_AD6450(*(_QWORD *)(a2 + 8));
            goto LABEL_124;
        }
      }
      sub_969240((__int64 *)&v67);
      break;
    case 6:
    case 0xE:
      v58 = 33;
LABEL_36:
      if ( v62 != 72 )
        goto LABEL_37;
      goto LABEL_70;
    case 7:
      v31 = sub_AD6400(*(_QWORD *)(a2 + 8));
      return sub_F162A0(a1, a2, v31);
    case 8:
      v31 = sub_AD6450(*(_QWORD *)(a2 + 8));
      return sub_F162A0(a1, a2, v31);
    default:
      BUG();
  }
  return v12;
}
