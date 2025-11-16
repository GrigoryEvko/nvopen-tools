// Function: sub_3516980
// Address: 0x3516980
//
char __fastcall sub_3516980(__int64 a1)
{
  unsigned __int64 v2; // rax
  __int64 *v3; // rax
  __int64 *v4; // rbx
  __int64 v5; // r13
  __int64 v6; // rax
  int v7; // ecx
  __int64 v8; // rsi
  int v9; // ecx
  unsigned int v10; // edx
  __int64 *v11; // rax
  __int64 v12; // rdi
  __int64 v13; // r12
  unsigned __int8 v14; // r14
  __int64 v15; // rax
  unsigned __int64 v16; // rax
  __int64 (*v17)(); // rax
  unsigned __int64 v18; // r12
  unsigned __int64 v19; // r12
  _QWORD *v20; // r14
  _QWORD *v21; // r12
  unsigned __int64 v22; // rsi
  _QWORD *v23; // rax
  _QWORD *v24; // rdi
  unsigned __int8 v25; // dl
  __int64 v26; // rcx
  __int64 v27; // rsi
  __int64 v28; // r10
  _QWORD *v29; // rcx
  _QWORD *v30; // r14
  _QWORD *v31; // rbx
  __int64 *v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rdx
  _BYTE *v35; // r13
  unsigned __int8 v36; // al
  _BYTE *v37; // r12
  _QWORD *v38; // r12
  char *v39; // rbx
  unsigned __int64 v40; // rsi
  char *v41; // rax
  char *v42; // rdi
  __int64 v43; // rcx
  __int64 v44; // rdx
  unsigned __int64 v45; // rdi
  __int64 v46; // rcx
  __int64 v47; // rdx
  int v48; // r12d
  char v49; // cl
  __int64 v50; // rdi
  __int64 v51; // rdi
  int v52; // r9d
  unsigned __int64 v53; // rdx
  unsigned __int64 v54; // rcx
  char v55; // si
  unsigned __int64 v56; // rcx
  __int64 v57; // r13
  __int64 v58; // r13
  __int64 i; // rbx
  int v60; // esi
  char v61; // dl
  char v62; // dl
  int v63; // eax
  int v64; // r8d
  unsigned __int8 v65; // al
  _BYTE *v66; // rsi
  __int64 v67; // rdx
  _QWORD *v68; // rax
  __int64 v69; // r12
  _QWORD *v70; // r14
  _QWORD *v71; // r12
  unsigned __int64 v72; // rsi
  _QWORD *v73; // rax
  _QWORD *v74; // rdi
  __int64 v75; // rax
  int v76; // esi
  _QWORD *v77; // rdi
  int v78; // eax
  unsigned __int64 v79; // rax
  __int64 v80; // rax
  __int64 v82; // [rsp+8h] [rbp-98h]
  __int64 *v83; // [rsp+10h] [rbp-90h]
  __int64 v84; // [rsp+18h] [rbp-88h]
  unsigned __int64 v85; // [rsp+20h] [rbp-80h]
  unsigned int v86; // [rsp+28h] [rbp-78h]
  unsigned __int8 v87; // [rsp+2Ch] [rbp-74h]
  unsigned int v88; // [rsp+2Ch] [rbp-74h]
  __int64 v89; // [rsp+30h] [rbp-70h]
  unsigned __int64 v90; // [rsp+38h] [rbp-68h]
  unsigned int v91; // [rsp+4Ch] [rbp-54h] BYREF
  __int64 v92; // [rsp+50h] [rbp-50h] BYREF
  unsigned __int64 v93; // [rsp+58h] [rbp-48h] BYREF
  __int64 v94; // [rsp+60h] [rbp-40h] BYREF
  __int64 v95[7]; // [rsp+68h] [rbp-38h] BYREF

  if ( !((unsigned int)qword_503CF08 | (unsigned int)qword_503CE28) )
  {
    LOBYTE(v2) = sub_B2D610(**(_QWORD **)(a1 + 520), 18);
    if ( (_BYTE)v2 )
      return v2;
    v69 = **(_QWORD **)(a1 + 520);
    if ( (unsigned __int8)sub_B2D610(v69, 47) || (unsigned __int8)sub_B2D610(v69, 18) )
    {
      v2 = *(_QWORD *)(**(_QWORD **)(a1 + 568) + 912LL);
      if ( (__int64 (*)())v2 == sub_2FE32F0 )
        return v2;
      LOBYTE(v2) = ((__int64 (*)(void))v2)();
      if ( !(_BYTE)v2 )
        return v2;
    }
  }
  v95[0] = *(_QWORD *)(*(_QWORD *)(a1 + 520) + 328LL);
  v2 = *sub_3515040(a1 + 888, v95);
  v90 = v2;
  if ( !*(_DWORD *)(v2 + 8) )
    return v2;
  sub_F02DB0(&v91, 1u, 5u);
  v92 = sub_2F06CB0(*(_QWORD *)(a1 + 536), *(_QWORD *)(*(_QWORD *)(a1 + 520) + 328LL));
  v85 = sub_1098D20((unsigned __int64 *)&v92, v91);
  v3 = *(__int64 **)v90;
  v89 = *(_QWORD *)v90 + 8LL * *(unsigned int *)(v90 + 8);
  if ( *(_QWORD *)v90 != v89 )
  {
    v4 = *(__int64 **)v90;
    while ( 1 )
    {
      v5 = *v4;
      if ( *v4 == *v3 )
        goto LABEL_16;
      v6 = *(_QWORD *)(a1 + 544);
      v7 = *(_DWORD *)(v6 + 24);
      v8 = *(_QWORD *)(v6 + 8);
      if ( !v7 )
        goto LABEL_16;
      v9 = v7 - 1;
      v10 = v9 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v11 = (__int64 *)(v8 + 16LL * v10);
      v12 = *v11;
      if ( v5 != *v11 )
      {
        v63 = 1;
        while ( v12 != -4096 )
        {
          v64 = v63 + 1;
          v10 = v9 & (v63 + v10);
          v11 = (__int64 *)(v8 + 16LL * v10);
          v12 = *v11;
          if ( v5 == *v11 )
            goto LABEL_9;
          v63 = v64;
        }
        goto LABEL_16;
      }
LABEL_9:
      v13 = v11[1];
      if ( !v13 )
        goto LABEL_16;
      v87 = (*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 568) + 896LL))(
              *(_QWORD *)(a1 + 568),
              v11[1]);
      v14 = v87;
      v15 = sub_2EA6730(v13);
      if ( v15 )
      {
        v25 = *(_BYTE *)(v15 - 16);
        if ( (v25 & 2) != 0 )
        {
          v26 = *(_QWORD *)(v15 - 32);
          v27 = *(unsigned int *)(v15 - 24);
        }
        else
        {
          v27 = (*(_WORD *)(v15 - 16) >> 6) & 0xF;
          v26 = v15 - 8LL * ((v25 >> 2) & 0xF) - 16;
        }
        v28 = v26 + 8 * v27;
        v29 = (_QWORD *)(v26 + 8);
        if ( (_QWORD *)v28 == v29 )
        {
          v14 = v87;
        }
        else
        {
          v83 = v4;
          v30 = v29;
          v31 = (_QWORD *)v28;
          v86 = 1;
          v84 = v5;
          v82 = v13;
          do
          {
            v35 = (_BYTE *)*v30;
            if ( (unsigned __int8)(*(_BYTE *)*v30 - 5) <= 0x1Fu )
            {
              v36 = *(v35 - 16);
              v37 = v35 - 16;
              v32 = (v36 & 2) != 0 ? (__int64 *)*((_QWORD *)v35 - 4) : (__int64 *)&v37[-8 * ((v36 >> 2) & 0xF)];
              if ( !*(_BYTE *)*v32 )
              {
                v33 = sub_B91420(*v32);
                if ( v34 == 15
                  && *(_QWORD *)v33 == 0x6F6F6C2E6D766C6CLL
                  && *(_DWORD *)(v33 + 8) == 1818308208
                  && *(_WORD *)(v33 + 12) == 26473
                  && *(_BYTE *)(v33 + 14) == 110 )
                {
                  v65 = *(v35 - 16);
                  if ( (v65 & 2) != 0 )
                    v66 = (_BYTE *)*((_QWORD *)v35 - 4);
                  else
                    v66 = &v37[-8 * ((v65 >> 2) & 0xF)];
                  v67 = *(_QWORD *)(*((_QWORD *)v66 + 1) + 136LL);
                  v68 = *(_QWORD **)(v67 + 24);
                  if ( *(_DWORD *)(v67 + 32) > 0x40u )
                    v68 = (_QWORD *)*v68;
                  v86 = (unsigned int)v68;
                }
              }
            }
            ++v30;
          }
          while ( v31 != v30 );
          v5 = v84;
          v4 = v83;
          v13 = v82;
          if ( !v86 )
          {
            v14 = -1;
            goto LABEL_12;
          }
          _BitScanReverse64(&v79, v86);
          v14 = 63 - (v79 ^ 0x3F);
          if ( v87 >= v14 )
            v14 = v87;
        }
      }
      if ( !v14 )
        goto LABEL_16;
LABEL_12:
      v93 = sub_2F06CB0(*(_QWORD *)(a1 + 536), v5);
      if ( v85 <= v93 )
      {
        v94 = sub_2F06CB0(*(_QWORD *)(a1 + 536), **(_QWORD **)(v13 + 32));
        v16 = sub_1098D20((unsigned __int64 *)&v94, v91);
        if ( v16 <= v93 )
        {
          if ( !sub_2EE6AD0(v5, *(_QWORD *)(a1 + 584), *(__int64 ***)(a1 + 536))
            || (v17 = *(__int64 (**)())(**(_QWORD **)(a1 + 568) + 912LL), v17 != sub_2FE32F0) && (unsigned __int8)v17() )
          {
            v18 = *(_QWORD *)v5 & 0xFFFFFFFFFFFFFFF8LL;
            if ( sub_2E322C0(v18, v5) )
            {
              v88 = sub_2E441D0(*(_QWORD *)(a1 + 528), v18, v5);
              v95[0] = sub_2F06CB0(*(_QWORD *)(a1 + 536), v18);
              v19 = sub_1098D20((unsigned __int64 *)v95, v88);
              if ( v19 > sub_1098D20(&v93, v91) )
                goto LABEL_16;
              *(_BYTE *)(v5 + 208) = v14;
              v20 = sub_C52410();
              v21 = v20 + 1;
              v22 = sub_C959E0();
              v23 = (_QWORD *)v20[2];
              if ( v23 )
              {
                v24 = v20 + 1;
                do
                {
                  if ( v22 > v23[4] )
                  {
                    v23 = (_QWORD *)v23[3];
                  }
                  else
                  {
                    v24 = v23;
                    v23 = (_QWORD *)v23[2];
                  }
                }
                while ( v23 );
                if ( v24 != v21 && v22 >= v24[4] )
                  v21 = v24;
              }
              if ( v21 == (_QWORD *)((char *)sub_C52410() + 8) )
                goto LABEL_117;
              v75 = v21[7];
              if ( !v75 )
                goto LABEL_117;
              v76 = dword_503CCC8;
              v77 = v21 + 6;
              do
              {
                if ( *(_DWORD *)(v75 + 32) < dword_503CCC8 )
                {
                  v75 = *(_QWORD *)(v75 + 24);
                }
                else
                {
                  v77 = (_QWORD *)v75;
                  v75 = *(_QWORD *)(v75 + 16);
                }
              }
              while ( v75 );
              if ( v77 == v21 + 6 )
              {
LABEL_117:
                v78 = (*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 568) + 904LL))(
                        *(_QWORD *)(a1 + 568),
                        v5);
                goto LABEL_118;
              }
            }
            else
            {
              *(_BYTE *)(v5 + 208) = v14;
              v70 = sub_C52410();
              v71 = v70 + 1;
              v72 = sub_C959E0();
              v73 = (_QWORD *)v70[2];
              if ( v73 )
              {
                v74 = v70 + 1;
                do
                {
                  if ( v72 > v73[4] )
                  {
                    v73 = (_QWORD *)v73[3];
                  }
                  else
                  {
                    v74 = v73;
                    v73 = (_QWORD *)v73[2];
                  }
                }
                while ( v73 );
                if ( v71 != v74 && v72 >= v74[4] )
                  v71 = v74;
              }
              if ( v71 == (_QWORD *)((char *)sub_C52410() + 8) )
                goto LABEL_117;
              v80 = v71[7];
              if ( !v80 )
                goto LABEL_117;
              v76 = dword_503CCC8;
              v77 = v71 + 6;
              do
              {
                if ( *(_DWORD *)(v80 + 32) < dword_503CCC8 )
                {
                  v80 = *(_QWORD *)(v80 + 24);
                }
                else
                {
                  v77 = (_QWORD *)v80;
                  v80 = *(_QWORD *)(v80 + 16);
                }
              }
              while ( v80 );
              if ( v71 + 6 == v77 )
                goto LABEL_117;
            }
            if ( v76 < *((_DWORD *)v77 + 8) )
              goto LABEL_117;
            v78 = qword_503CD48;
            if ( *((int *)v77 + 9) <= 0 )
              goto LABEL_117;
LABEL_118:
            *(_DWORD *)(v5 + 212) = v78;
          }
        }
      }
LABEL_16:
      if ( (__int64 *)v89 == ++v4 )
        break;
      v3 = *(__int64 **)v90;
    }
  }
  v38 = sub_C52410();
  v39 = (char *)(v38 + 1);
  v40 = sub_C959E0();
  v41 = (char *)v38[2];
  if ( v41 )
  {
    v42 = (char *)(v38 + 1);
    do
    {
      while ( 1 )
      {
        v43 = *((_QWORD *)v41 + 2);
        v44 = *((_QWORD *)v41 + 3);
        if ( v40 <= *((_QWORD *)v41 + 4) )
          break;
        v41 = (char *)*((_QWORD *)v41 + 3);
        if ( !v44 )
          goto LABEL_46;
      }
      v42 = v41;
      v41 = (char *)*((_QWORD *)v41 + 2);
    }
    while ( v43 );
LABEL_46:
    if ( v42 != v39 && v40 >= *((_QWORD *)v42 + 4) )
      v39 = v42;
  }
  v2 = (unsigned __int64)sub_C52410() + 8;
  if ( v39 == (char *)v2 || (v2 = *((_QWORD *)v39 + 7)) == 0 )
  {
    v48 = 0;
    goto LABEL_57;
  }
  v45 = (unsigned __int64)(v39 + 48);
  do
  {
    while ( 1 )
    {
      v46 = *(_QWORD *)(v2 + 16);
      v47 = *(_QWORD *)(v2 + 24);
      if ( *(_DWORD *)(v2 + 32) >= dword_503CCC8 )
        break;
      v2 = *(_QWORD *)(v2 + 24);
      if ( !v47 )
        goto LABEL_55;
    }
    v45 = v2;
    v2 = *(_QWORD *)(v2 + 16);
  }
  while ( v46 );
LABEL_55:
  v48 = 0;
  if ( (char *)v45 == v39 + 48 || dword_503CCC8 < *(_DWORD *)(v45 + 32) )
  {
LABEL_57:
    v49 = qword_503CF08;
    if ( (_DWORD)qword_503CF08 )
      goto LABEL_58;
    goto LABEL_70;
  }
  v49 = qword_503CF08;
  v48 = *(_DWORD *)(v45 + 36);
  if ( (_DWORD)qword_503CF08 )
  {
LABEL_58:
    v50 = *(_QWORD *)(a1 + 520);
    v2 = *(_QWORD *)(v50 + 328);
    v51 = v50 + 320;
    if ( v51 != v2 )
    {
      v52 = qword_503CD48;
      v53 = 1LL << v49;
      do
      {
        v55 = -1;
        if ( v48 > 0 )
        {
          if ( v53 )
          {
            _BitScanReverse64(&v54, v53);
            v55 = 63 - (v54 ^ 0x3F);
          }
          *(_BYTE *)(v2 + 208) = v55;
          *(_DWORD *)(v2 + 212) = v52;
        }
        else
        {
          if ( v53 )
          {
            _BitScanReverse64(&v56, v53);
            v55 = 63 - (v56 ^ 0x3F);
          }
          *(_BYTE *)(v2 + 208) = v55;
        }
        v2 = *(_QWORD *)(v2 + 8);
      }
      while ( v51 != v2 );
    }
    return v2;
  }
LABEL_70:
  if ( (_DWORD)qword_503CE28 )
  {
    v57 = *(_QWORD *)(a1 + 520);
    v2 = *(_QWORD *)(v57 + 328);
    v58 = v57 + 320;
    for ( i = *(_QWORD *)(v2 + 8); v58 != i; i = *(_QWORD *)(i + 8) )
    {
      LOBYTE(v2) = sub_2E322C0(*(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL, i);
      if ( !(_BYTE)v2 )
      {
        if ( v48 > 0 )
        {
          v60 = qword_503CD48;
          v61 = -1;
          v2 = 1LL << qword_503CE28;
          if ( 1LL << qword_503CE28 )
          {
            _BitScanReverse64(&v2, v2);
            LOBYTE(v2) = v2 ^ 0x3F;
            v61 = 63 - v2;
          }
          *(_BYTE *)(i + 208) = v61;
          *(_DWORD *)(i + 212) = v60;
        }
        else
        {
          v62 = -1;
          v2 = 1LL << qword_503CE28;
          if ( 1LL << qword_503CE28 )
          {
            _BitScanReverse64(&v2, v2);
            LOBYTE(v2) = v2 ^ 0x3F;
            v62 = 63 - v2;
          }
          *(_BYTE *)(i + 208) = v62;
        }
      }
    }
  }
  return v2;
}
