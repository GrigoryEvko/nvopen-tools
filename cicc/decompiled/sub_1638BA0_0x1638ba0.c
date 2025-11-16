// Function: sub_1638BA0
// Address: 0x1638ba0
//
__int64 __fastcall sub_1638BA0(__int64 *a1, __int64 a2, __int64 a3, __int64 *a4)
{
  void **v4; // rdx
  void **v5; // rax
  void **v6; // r12
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v10; // rdx
  __int64 v11; // rcx
  void **v12; // rdx
  void *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  int v16; // r9d
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rsi
  unsigned int i; // eax
  __int64 v21; // rsi
  unsigned int v22; // eax
  __int64 v23; // r8
  char v24; // al
  __int64 *v25; // rbx
  __int64 *v26; // r15
  __int64 v27; // r13
  __int64 v28; // r12
  char v29; // cl
  __int64 v30; // rdx
  int v31; // edi
  unsigned int v32; // esi
  __int64 v33; // rax
  __int64 v34; // r9
  __int64 v35; // rsi
  __int64 v36; // rax
  unsigned __int64 v37; // rdx
  _QWORD *v38; // rcx
  __int64 v39; // r12
  _QWORD *v40; // rbx
  _QWORD *v41; // rax
  _QWORD *v42; // rax
  __int64 v43; // r13
  _QWORD *v44; // rdx
  _QWORD *v45; // rdi
  _QWORD *v46; // rsi
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rcx
  int v51; // r10d
  unsigned __int64 v52; // rsi
  unsigned __int64 v53; // rsi
  unsigned int j; // eax
  _QWORD *v55; // rsi
  unsigned int v56; // eax
  int v57; // eax
  __int64 v58; // r8
  __int64 v59; // rbx
  __int64 v60; // rax
  int v61; // r8d
  __int64 v62; // [rsp+8h] [rbp-148h]
  __int64 v63; // [rsp+10h] [rbp-140h]
  __int64 v64; // [rsp+18h] [rbp-138h]
  bool v65; // [rsp+27h] [rbp-129h]
  void *v66; // [rsp+28h] [rbp-128h]
  __int64 v69; // [rsp+40h] [rbp-110h]
  __int64 *v70; // [rsp+48h] [rbp-108h]
  __int64 v73; // [rsp+60h] [rbp-F0h] BYREF
  char v74[8]; // [rsp+68h] [rbp-E8h] BYREF
  char v75[16]; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v76; // [rsp+80h] [rbp-D0h]
  char v77[8]; // [rsp+A0h] [rbp-B0h] BYREF
  _QWORD *v78; // [rsp+A8h] [rbp-A8h]
  _QWORD *v79; // [rsp+B0h] [rbp-A0h]
  int v80; // [rsp+B8h] [rbp-98h]
  int v81; // [rsp+BCh] [rbp-94h]
  int v82; // [rsp+C0h] [rbp-90h]
  char v83[16]; // [rsp+C8h] [rbp-88h] BYREF
  __int64 v84; // [rsp+D8h] [rbp-78h] BYREF
  _QWORD *v85; // [rsp+E0h] [rbp-70h]
  _QWORD *v86; // [rsp+E8h] [rbp-68h]
  unsigned int v87; // [rsp+F0h] [rbp-60h]
  unsigned int v88; // [rsp+F4h] [rbp-5Ch]
  int v89; // [rsp+F8h] [rbp-58h]
  char v90[16]; // [rsp+100h] [rbp-50h] BYREF
  char v91; // [rsp+110h] [rbp-40h]

  if ( *(_DWORD *)(a3 + 88) != *(_DWORD *)(a3 + 84) || !sub_134EB50(a3, (__int64)&unk_4F9EE48) )
  {
    v4 = *(void ***)(a3 + 72);
    v5 = *(void ***)(a3 + 64);
    v62 = a3 + 56;
    if ( v4 == v5 )
    {
      v6 = &v5[*(unsigned int *)(a3 + 84)];
      if ( v5 == v6 )
      {
        v12 = *(void ***)(a3 + 64);
      }
      else
      {
        do
        {
          if ( *v5 == &unk_4F9EE60 )
            break;
          ++v5;
        }
        while ( v6 != v5 );
        v12 = v6;
      }
LABEL_13:
      while ( v12 != v5 )
      {
        if ( (unsigned __int64)*v5 < 0xFFFFFFFFFFFFFFFELL )
          goto LABEL_6;
        ++v5;
      }
      if ( v6 != v5 )
        goto LABEL_7;
    }
    else
    {
      v6 = &v4[*(unsigned int *)(a3 + 80)];
      v5 = (void **)sub_16CC9F0(v62, &unk_4F9EE60);
      if ( *v5 == &unk_4F9EE60 )
      {
        v10 = *(_QWORD *)(a3 + 72);
        if ( v10 == *(_QWORD *)(a3 + 64) )
          v11 = *(unsigned int *)(a3 + 84);
        else
          v11 = *(unsigned int *)(a3 + 80);
        v12 = (void **)(v10 + 8 * v11);
        goto LABEL_13;
      }
      v7 = *(_QWORD *)(a3 + 72);
      if ( v7 == *(_QWORD *)(a3 + 64) )
      {
        v5 = (void **)(v7 + 8LL * *(unsigned int *)(a3 + 84));
        v12 = v5;
        goto LABEL_13;
      }
      v5 = (void **)(v7 + 8LL * *(unsigned int *)(a3 + 80));
LABEL_6:
      if ( v6 != v5 )
        goto LABEL_7;
    }
    if ( !sub_134EB50(a3, (__int64)&unk_4F9EE48)
      && !sub_134EB50(a3, (__int64)&unk_4F9EE60)
      && !sub_134EB50(a3, (__int64)&unk_4F9EE48)
      && !sub_134EB50(a3, (__int64)&unk_4F9EE70) )
    {
LABEL_7:
      v8 = *a1;
      sub_1636B20(*a1 + 64);
      sub_1636CF0(v8 + 32);
      return 1;
    }
    if ( *(_DWORD *)(a3 + 84) == *(_DWORD *)(a3 + 88) )
    {
      v65 = 1;
      if ( !sub_134EB50(a3, (__int64)&unk_4F9EE48) )
        v65 = sub_134EB50(a3, (__int64)&unk_4F9EE68);
    }
    else
    {
      v65 = 0;
    }
    v64 = a2 + 24;
    v69 = *(_QWORD *)(a2 + 32);
    if ( v69 == a2 + 24 )
      return 0;
    v63 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
    while ( 1 )
    {
      v91 = 0;
      v13 = (void *)(v69 - 56);
      if ( !v69 )
        v13 = 0;
      v66 = v13;
      v14 = *(unsigned int *)(*a1 + 88);
      if ( (_DWORD)v14 )
      {
        v15 = *(_QWORD *)(*a1 + 72);
        v16 = 1;
        v17 = (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)
             | ((unsigned __int64)(((unsigned int)&unk_4F9EE58 >> 9) ^ ((unsigned int)&unk_4F9EE58 >> 4)) << 32))
            - 1
            - ((unsigned __int64)(((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)) << 32);
        v18 = ((v17 >> 22) ^ v17) - 1 - (((v17 >> 22) ^ v17) << 13);
        v19 = ((9 * ((v18 >> 8) ^ v18)) >> 15) ^ (9 * ((v18 >> 8) ^ v18));
        for ( i = (v14 - 1) & (((v19 - 1 - (v19 << 27)) >> 31) ^ (v19 - 1 - ((_DWORD)v19 << 27))); ; i = (v14 - 1) & v22 )
        {
          v21 = v15 + 24LL * i;
          if ( *(_UNKNOWN **)v21 == &unk_4F9EE58 && v66 == *(void **)(v21 + 8) )
            break;
          if ( *(_QWORD *)v21 == -8 && *(_QWORD *)(v21 + 8) == -8 )
            goto LABEL_33;
          v22 = v16 + i;
          ++v16;
        }
        if ( v21 != v15 + 24 * v14 )
        {
          v23 = *(_QWORD *)(*(_QWORD *)(v21 + 16) + 24LL);
          if ( v23 )
            break;
        }
      }
LABEL_33:
      if ( !v65 )
      {
        sub_1638020(*a1, v66, a3);
        if ( v91 )
        {
LABEL_37:
          if ( v86 != v85 )
            _libc_free((unsigned __int64)v86);
          if ( v79 != v78 )
            _libc_free((unsigned __int64)v79);
        }
      }
LABEL_34:
      v69 = *(_QWORD *)(v69 + 8);
      if ( v64 == v69 )
        return 0;
    }
    v24 = *(_BYTE *)(v23 + 24) & 1;
    if ( *(_DWORD *)(v23 + 24) >> 1 )
    {
      if ( v24 )
      {
        v25 = (__int64 *)(v23 + 32);
        v26 = (__int64 *)(v23 + 64);
      }
      else
      {
        v25 = *(__int64 **)(v23 + 32);
        v58 = 2LL * *(unsigned int *)(v23 + 40);
        v26 = &v25[v58];
        if ( v25 == &v25[v58] )
          goto LABEL_50;
      }
      do
      {
        if ( *v25 != -16 && *v25 != -8 )
          break;
        v25 += 2;
      }
      while ( v25 != v26 );
    }
    else
    {
      if ( v24 )
      {
        v59 = v23 + 32;
        v60 = 32;
      }
      else
      {
        v59 = *(_QWORD *)(v23 + 32);
        v60 = 16LL * *(unsigned int *)(v23 + 40);
      }
      v25 = (__int64 *)(v60 + v59);
      v26 = v25;
    }
LABEL_50:
    if ( v26 == v25 )
      goto LABEL_33;
    while ( 1 )
    {
      v27 = *v25;
      v28 = *a4;
      v29 = *(_BYTE *)(*a4 + 8) & 1;
      if ( v29 )
      {
        v30 = v28 + 16;
        v31 = 7;
      }
      else
      {
        v36 = *(unsigned int *)(v28 + 24);
        v30 = *(_QWORD *)(v28 + 16);
        if ( !(_DWORD)v36 )
          goto LABEL_103;
        v31 = v36 - 1;
      }
      v32 = v31 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
      v33 = v30 + 16LL * v32;
      v34 = *(_QWORD *)v33;
      if ( v27 == *(_QWORD *)v33 )
        goto LABEL_54;
      v57 = 1;
      while ( v34 != -8 )
      {
        v61 = v57 + 1;
        v32 = v31 & (v57 + v32);
        v33 = v30 + 16LL * v32;
        v34 = *(_QWORD *)v33;
        if ( v27 == *(_QWORD *)v33 )
          goto LABEL_54;
        v57 = v61;
      }
      if ( v29 )
      {
        v47 = 128;
        goto LABEL_104;
      }
      v36 = *(unsigned int *)(v28 + 24);
LABEL_103:
      v47 = 16 * v36;
LABEL_104:
      v33 = v30 + v47;
LABEL_54:
      v35 = 128;
      if ( !v29 )
        v35 = 16LL * *(unsigned int *)(v28 + 24);
      if ( v33 == v35 + v30 )
      {
        v48 = a4[1];
        v49 = *(unsigned int *)(v48 + 24);
        v50 = *(_QWORD *)(v48 + 8);
        if ( (_DWORD)v49 )
        {
          v51 = 1;
          v52 = (((v63 | ((unsigned __int64)(((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4)) << 32))
                - 1
                - (v63 << 32)) >> 22)
              ^ ((v63 | ((unsigned __int64)(((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4)) << 32))
               - 1
               - (v63 << 32));
          v53 = ((9 * (((v52 - 1 - (v52 << 13)) >> 8) ^ (v52 - 1 - (v52 << 13)))) >> 15)
              ^ (9 * (((v52 - 1 - (v52 << 13)) >> 8) ^ (v52 - 1 - (v52 << 13))));
          for ( j = (v49 - 1) & (((v53 - 1 - (v53 << 27)) >> 31) ^ (v53 - 1 - ((_DWORD)v53 << 27))); ; j = (v49 - 1) & v56 )
          {
            v55 = (_QWORD *)(v50 + 24LL * j);
            if ( v27 == *v55 && a2 == v55[1] )
              break;
            if ( *v55 == -8 && v55[1] == -8 )
              goto LABEL_114;
            v56 = v51 + j;
            ++v51;
          }
        }
        else
        {
LABEL_114:
          v55 = (_QWORD *)(v50 + 24 * v49);
        }
        v74[0] = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64 *))(**(_QWORD **)(v55[2] + 24LL) + 16LL))(
                   *(_QWORD *)(v55[2] + 24LL),
                   a2,
                   a3,
                   a4);
        v73 = v27;
        sub_1367360((__int64)v75, v28, &v73, v74);
        v33 = v76;
      }
      if ( !*(_BYTE *)(v33 + 8) )
      {
        v25 += 2;
        goto LABEL_59;
      }
      if ( !v91 )
      {
        sub_16CCCB0(v77, v83, a3);
        sub_16CCCB0(&v84, v90, v62);
        v91 = 1;
      }
      v37 = v25[1] & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v25[1] & 4) != 0 )
      {
        v38 = *(_QWORD **)v37;
        v25 += 2;
        v39 = *(_QWORD *)v37 + 8LL * *(unsigned int *)(v37 + 8);
      }
      else
      {
        v38 = v25 + 1;
        v25 += 2;
        if ( !v37 )
          goto LABEL_59;
        v39 = (__int64)v25;
      }
      if ( (_QWORD *)v39 != v38 )
      {
        v70 = v25;
        v40 = v38;
        while ( 1 )
        {
LABEL_82:
          v43 = *v40;
          v41 = v78;
          if ( v79 == v78 )
          {
            v44 = &v78[v81];
            if ( v78 == v44 )
            {
LABEL_100:
              v41 = &v78[v81];
            }
            else
            {
              while ( v43 != *v41 )
              {
                if ( v44 == ++v41 )
                  goto LABEL_100;
              }
            }
          }
          else
          {
            v41 = (_QWORD *)sub_16CC9F0(v77, *v40);
            if ( v43 == *v41 )
            {
              if ( v79 == v78 )
                v44 = &v79[v81];
              else
                v44 = &v79[v80];
            }
            else
            {
              if ( v79 != v78 )
                goto LABEL_79;
              v41 = &v79[v81];
              v44 = v41;
            }
          }
          if ( v44 == v41 )
          {
LABEL_79:
            v42 = v85;
            if ( v86 != v85 )
              goto LABEL_80;
            goto LABEL_89;
          }
          *v41 = -2;
          v42 = v85;
          ++v82;
          if ( v86 != v85 )
            goto LABEL_80;
LABEL_89:
          v45 = &v42[v88];
          if ( v42 != v45 )
          {
            v46 = 0;
            while ( v43 != *v42 )
            {
              if ( *v42 == -2 )
                v46 = v42;
              if ( v45 == ++v42 )
              {
                if ( !v46 )
                  goto LABEL_110;
                ++v40;
                *v46 = v43;
                --v89;
                ++v84;
                if ( (_QWORD *)v39 != v40 )
                  goto LABEL_82;
                goto LABEL_97;
              }
            }
            goto LABEL_81;
          }
LABEL_110:
          if ( v88 < v87 )
          {
            ++v88;
            *v45 = v43;
            ++v84;
            goto LABEL_81;
          }
LABEL_80:
          sub_16CCBA0(&v84, v43);
LABEL_81:
          if ( (_QWORD *)v39 == ++v40 )
          {
LABEL_97:
            v25 = v70;
            break;
          }
        }
      }
LABEL_59:
      if ( v25 != v26 )
      {
        while ( *v25 == -16 || *v25 == -8 )
        {
          v25 += 2;
          if ( v26 == v25 )
            goto LABEL_63;
        }
        if ( v26 != v25 )
          continue;
      }
LABEL_63:
      if ( v91 )
      {
        sub_1638020(*a1, v66, (__int64)v77);
        if ( v91 )
          goto LABEL_37;
        goto LABEL_34;
      }
      goto LABEL_33;
    }
  }
  return 0;
}
