// Function: sub_223C8C0
// Address: 0x223c8c0
//
_QWORD *__fastcall sub_223C8C0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        _QWORD *a4,
        __int64 a5,
        __int64 a6,
        _DWORD *a7,
        __int64 *a8)
{
  _QWORD *v8; // rbp
  __int64 v10; // r12
  __int64 *v11; // rbx
  __int64 v12; // r14
  bool v13; // zf
  int v14; // r15d
  __int64 v15; // rbp
  __int64 v16; // rbp
  unsigned __int64 i; // rbx
  _QWORD *v18; // rdi
  _BYTE *v19; // rax
  bool v20; // al
  int v21; // eax
  int v22; // eax
  _QWORD *v23; // rdi
  char v24; // r13
  __int64 v25; // rbp
  volatile signed __int32 *v26; // rax
  unsigned __int64 v27; // rax
  char v28; // bl
  char v29; // bp
  char v30; // al
  char v31; // bp
  char v32; // bl
  _BYTE *v33; // rax
  __int64 v34; // rbp
  volatile signed __int32 *v35; // rax
  char v36; // bl
  unsigned __int64 v37; // rbp
  _QWORD *v38; // rdi
  _BYTE *v39; // rax
  int v40; // eax
  int v41; // eax
  _QWORD *v42; // rdi
  char v43; // al
  char v44; // r13
  int v45; // eax
  unsigned __int8 *v46; // rax
  char v47; // bl
  char v48; // r13
  int v49; // r8d
  char *v50; // rax
  bool v51; // al
  _QWORD *v52; // r12
  volatile signed __int32 *v53; // rdi
  unsigned __int64 v54; // rdi
  int v56; // r8d
  char v57; // si
  unsigned __int64 v58; // rax
  unsigned __int64 v59; // rdx
  int v60; // eax
  __int64 v61; // r13
  volatile signed __int32 *v62; // rax
  int v63; // eax
  int v64; // eax
  __int64 v65; // [rsp+8h] [rbp-D0h]
  unsigned __int64 v66; // [rsp+20h] [rbp-B8h]
  __int64 v67; // [rsp+30h] [rbp-A8h]
  unsigned __int64 v68; // [rsp+48h] [rbp-90h]
  char v70; // [rsp+58h] [rbp-80h]
  char v71; // [rsp+5Dh] [rbp-7Bh]
  char v72; // [rsp+5Eh] [rbp-7Ah]
  bool v73; // [rsp+5Fh] [rbp-79h]
  _QWORD *v74; // [rsp+60h] [rbp-78h] BYREF
  __int64 v75; // [rsp+68h] [rbp-70h]
  _QWORD *v76; // [rsp+70h] [rbp-68h] BYREF
  __int64 v77; // [rsp+78h] [rbp-60h]
  int v78; // [rsp+8Ch] [rbp-4Ch]
  volatile signed __int32 *v79; // [rsp+90h] [rbp-48h] BYREF
  volatile signed __int32 *v80[8]; // [rsp+98h] [rbp-40h] BYREF

  v8 = (_QWORD *)(a6 + 208);
  v76 = (_QWORD *)a2;
  v77 = a3;
  v74 = a4;
  v75 = a5;
  v67 = sub_222F790((_QWORD *)(a6 + 208), a2);
  v10 = sub_22091A0(&qword_4FD69D8);
  v11 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a6 + 208) + 24LL) + 8 * v10);
  v12 = *v11;
  if ( !*v11 )
  {
    v61 = sub_22077B0(0x70u);
    *(_DWORD *)(v61 + 8) = 0;
    *(_QWORD *)(v61 + 16) = 0;
    *(_QWORD *)(v61 + 24) = 0;
    *(_WORD *)(v61 + 32) = 0;
    *(_QWORD *)v61 = off_4A04880;
    *(_BYTE *)(v61 + 34) = 0;
    *(_QWORD *)(v61 + 40) = 0;
    *(_QWORD *)(v61 + 48) = 0;
    *(_QWORD *)(v61 + 56) = 0;
    *(_QWORD *)(v61 + 64) = 0;
    *(_QWORD *)(v61 + 72) = 0;
    *(_QWORD *)(v61 + 80) = 0;
    *(_QWORD *)(v61 + 88) = 0;
    *(_DWORD *)(v61 + 96) = 0;
    *(_BYTE *)(v61 + 111) = 0;
    sub_2230AE0(v61, v8);
    sub_2209690(*(_QWORD *)(a6 + 208), (volatile signed __int32 *)v61, v10);
    v12 = *v11;
  }
  if ( *(_QWORD *)(v12 + 64) )
    v73 = *(_QWORD *)(v12 + 80) != 0;
  else
    v73 = 0;
  v13 = *(_BYTE *)(v12 + 32) == 0;
  v79 = (volatile signed __int32 *)&unk_4FD67D8;
  if ( !v13 )
    sub_2215AB0((__int64 *)&v79, 0x20u);
  v80[0] = (volatile signed __int32 *)&unk_4FD67D8;
  sub_2215AB0((__int64 *)v80, 0x20u);
  v71 = 0;
  v14 = 0;
  v65 = 0;
  v78 = *(_DWORD *)(v12 + 96);
  v70 = 0;
  v66 = 0;
  v72 = 0;
  while ( 2 )
  {
    switch ( *((_BYTE *)&v78 + v65) )
    {
      case 0:
        LOBYTE(v15) = 1;
        goto LABEL_10;
      case 1:
        if ( !sub_2233E50((__int64)&v76, (__int64)&v74)
          && (*(_BYTE *)(*(_QWORD *)(v67 + 48) + 2LL * (unsigned __int8)sub_2233F00((__int64)&v76) + 1) & 0x20) != 0 )
        {
          sub_22408B0(v76);
          LODWORD(v77) = -1;
          LOBYTE(v15) = 1;
LABEL_10:
          if ( v65 == 3 )
            goto LABEL_11;
          goto LABEL_77;
        }
        LOBYTE(v15) = 0;
        if ( v65 != 3 )
        {
LABEL_77:
          v41 = v77;
          v42 = v76;
          while ( 1 )
          {
            v47 = v41 == -1;
            v48 = v47 & (v42 != 0);
            if ( v48 )
            {
              v47 = 0;
              if ( v42[2] >= v42[3] && (*(unsigned int (__fastcall **)(_QWORD *))(*v42 + 72LL))(v42) == -1 )
              {
                v76 = 0;
                v47 = v48;
              }
            }
            v43 = (_DWORD)v75 == -1;
            v44 = v43 & (v74 != 0);
            if ( v44 )
            {
              v43 = 0;
              if ( v74[2] >= v74[3] )
              {
                v56 = (*(__int64 (**)(void))(*v74 + 72LL))();
                v43 = 0;
                if ( v56 == -1 )
                {
                  v74 = 0;
                  v43 = v44;
                }
              }
            }
            if ( v43 == v47 )
              goto LABEL_111;
            LOBYTE(v45) = v77;
            v42 = v76;
            if ( (_DWORD)v77 == -1 && v76 )
            {
              v46 = (unsigned __int8 *)v76[2];
              if ( (unsigned __int64)v46 < v76[3] )
              {
                if ( (*(_BYTE *)(*(_QWORD *)(v67 + 48) + 2LL * *v46 + 1) & 0x20) == 0 )
                  goto LABEL_111;
LABEL_84:
                v42[2] = v46 + 1;
                goto LABEL_85;
              }
              v45 = (*(__int64 (**)(void))(*v76 + 72LL))();
              if ( v45 == -1 )
                v76 = 0;
            }
            if ( (*(_BYTE *)(*(_QWORD *)(v67 + 48) + 2LL * (unsigned __int8)v45 + 1) & 0x20) == 0 )
              goto LABEL_111;
            v42 = v76;
            v46 = (unsigned __int8 *)v76[2];
            if ( (unsigned __int64)v46 < v76[3] )
              goto LABEL_84;
            (*(void (__fastcall **)(_QWORD *))(*v76 + 80LL))(v76);
            v42 = v76;
LABEL_85:
            LODWORD(v77) = -1;
            v41 = -1;
          }
        }
LABEL_11:
        if ( ((unsigned __int8)v15 & (v66 > 1)) == 0 )
        {
          if ( (_BYTE)v15 )
          {
            if ( *((_QWORD *)v80[0] - 3) > 1u )
              goto LABEL_146;
            goto LABEL_134;
          }
          goto LABEL_103;
        }
        if ( v72 )
          v16 = *(_QWORD *)(v12 + 72);
        else
          v16 = *(_QWORD *)(v12 + 56);
        for ( i = 1; ; ++i )
        {
          v20 = sub_2233E50((__int64)&v76, (__int64)&v74);
          if ( i >= v66 || v20 )
            break;
          v18 = v76;
          LOBYTE(v21) = v77;
          if ( v76 && (_DWORD)v77 == -1 )
          {
            v19 = (_BYTE *)v76[2];
            if ( (unsigned __int64)v19 < v76[3] )
            {
              if ( *v19 != *(_BYTE *)(v16 + i) )
                goto LABEL_103;
LABEL_17:
              v18[2] = v19 + 1;
              goto LABEL_18;
            }
            v21 = (*(__int64 (**)(void))(*v76 + 72LL))();
            if ( v21 == -1 )
              v76 = 0;
          }
          if ( *(_BYTE *)(v16 + i) != (_BYTE)v21 )
            goto LABEL_103;
          v18 = v76;
          v19 = (_BYTE *)v76[2];
          if ( (unsigned __int64)v19 < v76[3] )
            goto LABEL_17;
          (*(void (__fastcall **)(_QWORD *))(*v76 + 80LL))(v76);
LABEL_18:
          LODWORD(v77) = -1;
        }
        if ( i != v66 )
          goto LABEL_103;
        if ( *((_QWORD *)v80[0] - 3) <= 1u )
          goto LABEL_134;
LABEL_146:
        v58 = sub_22153C0(v80, 48, 0);
        if ( v58 )
        {
          v59 = *((_QWORD *)v80[0] - 3);
          if ( v58 == -1 )
            v58 = v59 - 1;
          if ( v58 <= v59 )
            v59 = v58;
          sub_2215540(v80, 0, v59, 0);
        }
LABEL_134:
        if ( v72 )
        {
          v62 = v80[0];
          if ( *((int *)v80[0] - 2) >= 0 )
          {
            sub_2215730(v80);
            v62 = v80[0];
          }
          if ( *(_BYTE *)v62 != 48 )
          {
            sub_2215790(v80);
            sub_22157B0(v80, 0, 0, 1u, 45);
            *((_DWORD *)v80[0] - 2) = -1;
          }
        }
        if ( *((_QWORD *)v79 - 3) )
        {
          v57 = v14;
          if ( v71 )
            v57 = v70;
          sub_2215DF0((__int64 *)&v79, v57);
          if ( !(unsigned __int8)sub_2255280(*(_QWORD *)(v12 + 16), *(_QWORD *)(v12 + 24), &v79) )
            *a7 |= 4u;
        }
        if ( v71 && *(_DWORD *)(v12 + 88) != v14 )
          goto LABEL_103;
        sub_2215390(a8, (__int64 *)v80);
        v51 = sub_2233E50((__int64)&v76, (__int64)&v74);
LABEL_104:
        if ( v51 )
          *a7 |= 2u;
        v52 = v76;
        v53 = v80[0] - 6;
        if ( v80[0] - 6 != (volatile signed __int32 *)&unk_4FD67C0 )
        {
          if ( &_pthread_key_create )
          {
            v64 = _InterlockedExchangeAdd(v80[0] - 2, 0xFFFFFFFF);
          }
          else
          {
            v64 = *((_DWORD *)v80[0] - 2);
            *((_DWORD *)v80[0] - 2) = v64 - 1;
          }
          if ( v64 <= 0 )
            j_j___libc_free_0_1((unsigned __int64)v53);
        }
        v54 = (unsigned __int64)(v79 - 6);
        if ( v79 - 6 != (volatile signed __int32 *)&unk_4FD67C0 )
        {
          if ( &_pthread_key_create )
          {
            v63 = _InterlockedExchangeAdd(v79 - 2, 0xFFFFFFFF);
          }
          else
          {
            v63 = *((_DWORD *)v79 - 2);
            *((_DWORD *)v79 - 2) = v63 - 1;
          }
          if ( v63 <= 0 )
            j_j___libc_free_0_1(v54);
        }
        return v52;
      case 2:
        if ( (*(_BYTE *)(a6 + 25) & 2) != 0 )
          goto LABEL_63;
        v36 = (_DWORD)v65 == 0 || v66 > 1;
        if ( v36 )
          goto LABEL_63;
        if ( (_DWORD)v65 == 1 )
        {
          if ( v73 || (_BYTE)v78 == 3 || BYTE2(v78) == 1 )
          {
LABEL_63:
            v37 = 0;
            v68 = *(_QWORD *)(v12 + 48);
            while ( 1 )
            {
              v36 = !sub_2233E50((__int64)&v76, (__int64)&v74) && v37 < v68;
              if ( !v36 )
              {
                if ( v37 == v68 )
                  break;
LABEL_73:
                if ( !v37 )
                {
                  v15 = (*(_BYTE *)(a6 + 25) & 2) == 0;
                  v36 = (*(_BYTE *)(a6 + 25) & 2) != 0;
                  goto LABEL_112;
                }
LABEL_103:
                *a7 |= 4u;
                v51 = sub_2233E50((__int64)&v76, (__int64)&v74);
                goto LABEL_104;
              }
              v38 = v76;
              LOBYTE(v40) = v77;
              if ( v76 && (_DWORD)v77 == -1 )
              {
                v39 = (_BYTE *)v76[2];
                if ( (unsigned __int64)v39 < v76[3] )
                {
                  if ( *(_BYTE *)(*(_QWORD *)(v12 + 40) + v37) != *v39 )
                    goto LABEL_73;
LABEL_66:
                  v38[2] = v39 + 1;
                  goto LABEL_67;
                }
                v40 = (*(__int64 (**)(void))(*v76 + 72LL))();
                if ( v40 == -1 )
                  v76 = 0;
              }
              if ( *(_BYTE *)(*(_QWORD *)(v12 + 40) + v37) != (_BYTE)v40 )
                goto LABEL_73;
              v38 = v76;
              v39 = (_BYTE *)v76[2];
              if ( (unsigned __int64)v39 < v76[3] )
                goto LABEL_66;
              (*(void (__fastcall **)(_QWORD *))(*v76 + 80LL))(v76);
LABEL_67:
              LODWORD(v77) = -1;
              ++v37;
            }
          }
LABEL_145:
          LOBYTE(v15) = 1;
          goto LABEL_112;
        }
        LOBYTE(v15) = 1;
        if ( (_DWORD)v65 == 2 )
        {
          if ( HIBYTE(v78) == 4 || HIBYTE(v78) == 3 && v73 )
            goto LABEL_63;
        }
        else
        {
LABEL_112:
          if ( (int)v65 + 1 > 3 || v36 )
            goto LABEL_11;
        }
        ++v65;
        continue;
      case 3:
        if ( *(_QWORD *)(v12 + 64) )
        {
          v36 = sub_2233E50((__int64)&v76, (__int64)&v74);
          if ( !v36 && **(_BYTE **)(v12 + 56) == (unsigned __int8)sub_2233F00((__int64)&v76) )
          {
            v66 = *(_QWORD *)(v12 + 64);
            sub_22408B0(v76);
            LODWORD(v77) = -1;
            goto LABEL_145;
          }
          if ( !*(_QWORD *)(v12 + 80) )
          {
            if ( !*(_QWORD *)(v12 + 64) )
              goto LABEL_55;
            goto LABEL_123;
          }
        }
        else if ( !*(_QWORD *)(v12 + 80) )
        {
          goto LABEL_55;
        }
        v36 = sub_2233E50((__int64)&v76, (__int64)&v74);
        if ( !v36 && **(_BYTE **)(v12 + 72) == (unsigned __int8)sub_2233F00((__int64)&v76) )
        {
          v66 = *(_QWORD *)(v12 + 80);
          sub_22408B0(v76);
          LODWORD(v77) = -1;
          LOBYTE(v15) = 1;
          v72 = 1;
          goto LABEL_112;
        }
        if ( !*(_QWORD *)(v12 + 64) || *(_QWORD *)(v12 + 80) )
        {
LABEL_55:
          v36 = v73;
          LOBYTE(v15) = !v73;
          goto LABEL_112;
        }
LABEL_123:
        v72 = 1;
        v36 = 0;
        LOBYTE(v15) = 1;
        goto LABEL_112;
      case 4:
        v22 = v77;
        v23 = v76;
        while ( 2 )
        {
          v28 = v22 == -1;
          v29 = v28 & (v23 != 0);
          if ( v29 )
          {
            v28 = 0;
            if ( v23[2] >= v23[3] && (*(unsigned int (__fastcall **)(_QWORD *))(*v23 + 72LL))(v23) == -1 )
            {
              v76 = 0;
              v28 = v29;
            }
          }
          v30 = (_DWORD)v75 == -1;
          v31 = v30 & (v74 != 0);
          if ( v31 )
          {
            v30 = 0;
            if ( v74[2] >= v74[3] )
            {
              v49 = (*(__int64 (**)(void))(*v74 + 72LL))();
              v30 = 0;
              if ( v49 == -1 )
              {
                v74 = 0;
                v30 = v31;
              }
            }
          }
          if ( v28 == v30 )
            goto LABEL_101;
          v32 = v77;
          if ( v76 && (_DWORD)v77 == -1 )
          {
            v50 = (char *)v76[2];
            if ( (unsigned __int64)v50 >= v76[3] )
            {
              v60 = (*(__int64 (**)(void))(*v76 + 72LL))();
              v32 = v60;
              if ( v60 == -1 )
              {
                v76 = 0;
                v32 = -1;
              }
            }
            else
            {
              v32 = *v50;
            }
          }
          v33 = memchr((const void *)(v12 + 101), v32, 0xAu);
          if ( !v33 )
          {
            LOBYTE(v15) = v71 | (*(_BYTE *)(v12 + 33) != (unsigned __int8)v32);
            if ( (_BYTE)v15 )
            {
              if ( *(_BYTE *)(v12 + 32) )
              {
                if ( *(_BYTE *)(v12 + 34) == v32 )
                {
                  if ( v71 )
                  {
                    LOBYTE(v15) = v71;
                  }
                  else
                  {
                    if ( v14 )
                    {
                      v34 = *((_QWORD *)v79 - 3);
                      if ( (unsigned __int64)(v34 + 1) > *((_QWORD *)v79 - 2) || *((int *)v79 - 2) > 0 )
                        sub_2215AB0((__int64 *)&v79, v34 + 1);
                      *((_BYTE *)v79 + *((_QWORD *)v79 - 3)) = v14;
                      v35 = v79;
                      v14 = 0;
                      if ( v79 - 6 != (volatile signed __int32 *)&unk_4FD67C0 )
                      {
                        *((_DWORD *)v79 - 2) = 0;
                        *((_QWORD *)v35 - 3) = v34 + 1;
                        *((_BYTE *)v35 + v34 + 1) = 0;
                        v23 = v76;
                        v27 = v76[2];
                        if ( v27 >= v76[3] )
                          goto LABEL_52;
                        goto LABEL_34;
                      }
LABEL_33:
                      v23 = v76;
                      v27 = v76[2];
                      if ( v27 >= v76[3] )
                      {
LABEL_52:
                        (*(void (__fastcall **)(_QWORD *))(*v23 + 80LL))(v23);
                        v23 = v76;
                        goto LABEL_35;
                      }
LABEL_34:
                      v23[2] = v27 + 1;
LABEL_35:
                      LODWORD(v77) = -1;
                      v22 = -1;
                      continue;
                    }
                    LOBYTE(v15) = 0;
                  }
                }
                else
                {
                  LOBYTE(v15) = *(_BYTE *)(v12 + 32);
                }
              }
            }
            else
            {
              if ( *(int *)(v12 + 88) > 0 )
              {
                v70 = v14;
                v14 = 0;
                v71 = 1;
                goto LABEL_33;
              }
              v71 = 0;
LABEL_101:
              LOBYTE(v15) = 1;
            }
            if ( *((_QWORD *)v80[0] - 3) )
            {
LABEL_111:
              v36 = v15 ^ 1;
              goto LABEL_112;
            }
            goto LABEL_103;
          }
          break;
        }
        v24 = v33[(_QWORD)off_4CDFAD0 - 100 - v12];
        v25 = *((_QWORD *)v80[0] - 3);
        if ( (unsigned __int64)(v25 + 1) > *((_QWORD *)v80[0] - 2) || *((int *)v80[0] - 2) > 0 )
          sub_2215AB0((__int64 *)v80, v25 + 1);
        *((_BYTE *)v80[0] + *((_QWORD *)v80[0] - 3)) = v24;
        v26 = v80[0];
        if ( v80[0] - 6 != (volatile signed __int32 *)&unk_4FD67C0 )
        {
          *((_DWORD *)v80[0] - 2) = 0;
          *((_QWORD *)v26 - 3) = v25 + 1;
          *((_BYTE *)v26 + v25 + 1) = 0;
        }
        ++v14;
        goto LABEL_33;
      default:
        v36 = 0;
        goto LABEL_145;
    }
  }
}
