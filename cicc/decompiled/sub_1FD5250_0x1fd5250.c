// Function: sub_1FD5250
// Address: 0x1fd5250
//
void __fastcall sub_1FD5250(_QWORD *a1, unsigned __int8 *a2, unsigned int a3, __int64 a4)
{
  __int64 v7; // rbx
  __int64 v8; // rcx
  int v9; // eax
  int v10; // esi
  __int64 v11; // r8
  unsigned int v12; // eax
  int v13; // edx
  int v14; // edi
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned int v20; // eax
  __int64 *v21; // r15
  unsigned int v22; // r11d
  __int64 v23; // r14
  __int64 *v24; // rdx
  __int64 v25; // r9
  __int64 v26; // rcx
  unsigned int v27; // esi
  __int64 *v28; // r8
  unsigned int v29; // ecx
  __int64 v30; // rbx
  __int64 v31; // r14
  unsigned __int64 *v32; // rcx
  unsigned __int64 v33; // rdx
  __int64 v34; // rsi
  __int64 v35; // r14
  __int64 v36; // rsi
  unsigned __int8 *v37; // rsi
  _BYTE *v38; // rdi
  _QWORD **v39; // rbx
  _QWORD **v40; // r14
  _QWORD *v41; // r12
  unsigned __int64 *v42; // rcx
  unsigned __int64 v43; // rdx
  __int64 v44; // rax
  unsigned int v45; // esi
  __int64 v46; // r9
  unsigned int v47; // edx
  __int64 v48; // rax
  __int64 v49; // r8
  unsigned int v50; // eax
  __int64 v51; // rax
  int v52; // ecx
  int v53; // edi
  __int64 v54; // rcx
  int v55; // ecx
  int v56; // edx
  __int64 v57; // rcx
  unsigned __int64 v58; // rax
  int v59; // [rsp+0h] [rbp-80h]
  __int64 v60; // [rsp+8h] [rbp-78h]
  __int64 v61; // [rsp+8h] [rbp-78h]
  __int64 v62; // [rsp+8h] [rbp-78h]
  char v63; // [rsp+14h] [rbp-6Ch]
  unsigned int v64; // [rsp+14h] [rbp-6Ch]
  __int64 v65; // [rsp+18h] [rbp-68h]
  int v66; // [rsp+18h] [rbp-68h]
  __int64 v67; // [rsp+18h] [rbp-68h]
  __int64 v68; // [rsp+20h] [rbp-60h] BYREF
  unsigned __int8 *v69; // [rsp+28h] [rbp-58h] BYREF
  _QWORD *v70; // [rsp+30h] [rbp-50h] BYREF
  __int64 v71; // [rsp+38h] [rbp-48h]
  _BYTE v72[64]; // [rsp+40h] [rbp-40h] BYREF

  v7 = a3;
  v8 = a1[5];
  v9 = *(_DWORD *)(v8 + 560);
  if ( v9 )
  {
    v10 = v9 - 1;
    v11 = *(_QWORD *)(v8 + 544);
    v12 = (v9 - 1) & (37 * a3);
    v13 = *(_DWORD *)(v11 + 4LL * v12);
    if ( v13 == (_DWORD)v7 )
      return;
    v14 = 1;
    while ( v13 != -1 )
    {
      v12 = v10 & (v14 + v12);
      v13 = *(_DWORD *)(v11 + 4LL * v12);
      if ( v13 == (_DWORD)v7 )
        return;
      ++v14;
    }
  }
  v15 = *(_QWORD *)(v8 + 904);
  v16 = *(_QWORD *)(v8 + 912);
  if ( v16 != v15 )
  {
    while ( *(_DWORD *)(v15 + 8) != (_DWORD)v7 )
    {
      v15 += 16;
      if ( v16 == v15 )
        goto LABEL_7;
    }
    v63 = 1;
LABEL_20:
    if ( !*(_DWORD *)(a4 + 16) )
    {
      v67 = a4;
      sub_1FD4EF0(a4, *(_QWORD *)(v8 + 784), a1[20]);
      a4 = v67;
    }
    v60 = a1[7];
    if ( (int)v7 < 0 )
      v19 = *(_QWORD *)(*(_QWORD *)(a1[7] + 24LL) + 16 * (v7 & 0x7FFFFFFF) + 8);
    else
      v19 = *(_QWORD *)(*(_QWORD *)(v60 + 272) + 8LL * (unsigned int)v7);
    while ( v19 )
    {
      if ( (*(_BYTE *)(v19 + 3) & 0x10) == 0 && (*(_BYTE *)(v19 + 4) & 8) == 0 )
      {
        v23 = *(_QWORD *)(a4 + 8);
        v22 = -1;
        v21 = 0;
        v24 = *(__int64 **)(v19 + 16);
        v25 = *(unsigned int *)(a4 + 24);
        v66 = v25 - 1;
LABEL_32:
        v26 = v23 + 16 * v25;
        if ( (_DWORD)v25 )
        {
          v27 = v66 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
          v26 = v23 + 16LL * v27;
          v28 = *(__int64 **)v26;
          if ( *(__int64 **)v26 != v24 )
          {
            v52 = 1;
            while ( v28 != (__int64 *)-8LL )
            {
              v27 = v66 & (v52 + v27);
              v59 = v52 + 1;
              v26 = v23 + 16LL * v27;
              v28 = *(__int64 **)v26;
              if ( *(__int64 **)v26 == v24 )
                goto LABEL_34;
              v52 = v59;
            }
            v26 = v23 + 16 * v25;
          }
        }
LABEL_34:
        v29 = *(_DWORD *)(v26 + 8);
        if ( v29 < v22 )
        {
          v22 = v29;
          v21 = v24;
        }
        while ( 1 )
        {
          v19 = *(_QWORD *)(v19 + 32);
          if ( !v19 )
            break;
          if ( (*(_BYTE *)(v19 + 3) & 0x10) == 0 && (*(_BYTE *)(v19 + 4) & 8) == 0 && *(__int64 **)(v19 + 16) != v24 )
          {
            v24 = *(__int64 **)(v19 + 16);
            goto LABEL_32;
          }
        }
        if ( !v63 || (v20 = *(_DWORD *)(a4 + 40), v20 >= v22) )
        {
          if ( v21 )
            goto LABEL_44;
          goto LABEL_85;
        }
LABEL_30:
        v21 = *(__int64 **)(a4 + 32);
        v22 = v20;
        goto LABEL_44;
      }
      v19 = *(_QWORD *)(v19 + 32);
    }
    if ( v63 )
    {
      v20 = *(_DWORD *)(a4 + 40);
      if ( v20 != -1 )
        goto LABEL_30;
    }
    v22 = -1;
LABEL_85:
    v21 = (__int64 *)(*(_QWORD *)(a1[5] + 784LL) + 24LL);
LABEL_44:
    v70 = v72;
    v71 = 0x100000000LL;
    if ( (int)v7 < 0 )
      v30 = *(_QWORD *)(*(_QWORD *)(v60 + 24) + 16 * (v7 & 0x7FFFFFFF) + 8);
    else
      v30 = *(_QWORD *)(*(_QWORD *)(v60 + 272) + 8 * v7);
    if ( v30 )
    {
      if ( (*(_BYTE *)(v30 + 3) & 0x10) == 0 )
      {
LABEL_48:
        v31 = *(_QWORD *)(v30 + 16);
        v64 = v22;
        if ( **(_WORD **)(v31 + 16) != 12 )
          goto LABEL_51;
LABEL_69:
        v45 = *(_DWORD *)(a4 + 24);
        v68 = v31;
        if ( v45 )
        {
          v46 = *(_QWORD *)(a4 + 8);
          v47 = (v45 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
          v48 = v46 + 16LL * v47;
          v49 = *(_QWORD *)v48;
          if ( *(_QWORD *)v48 == v31 )
          {
LABEL_71:
            v50 = *(_DWORD *)(v48 + 8);
            goto LABEL_72;
          }
          v53 = 1;
          v54 = 0;
          while ( v49 != -8 )
          {
            if ( v49 == -16 && !v54 )
              v54 = v48;
            v47 = (v45 - 1) & (v53 + v47);
            v48 = v46 + 16LL * v47;
            v49 = *(_QWORD *)v48;
            if ( v31 == *(_QWORD *)v48 )
              goto LABEL_71;
            ++v53;
          }
          if ( v54 )
            v48 = v54;
          v55 = *(_DWORD *)(a4 + 16);
          ++*(_QWORD *)a4;
          v56 = v55 + 1;
          if ( 4 * (v55 + 1) < 3 * v45 )
          {
            v57 = v31;
            LODWORD(v49) = v45 >> 3;
            if ( v45 - *(_DWORD *)(a4 + 20) - v56 > v45 >> 3 )
            {
LABEL_93:
              *(_DWORD *)(a4 + 16) = v56;
              if ( *(_QWORD *)v48 != -8 )
                --*(_DWORD *)(a4 + 20);
              *(_QWORD *)v48 = v57;
              *(_DWORD *)(v48 + 8) = 0;
              v50 = 0;
LABEL_72:
              if ( v64 > v50 )
              {
                v51 = (unsigned int)v71;
                if ( (unsigned int)v71 >= HIDWORD(v71) )
                {
                  v62 = a4;
                  sub_16CD150((__int64)&v70, v72, 0, 8, v49, v46);
                  v51 = (unsigned int)v71;
                  a4 = v62;
                }
                v70[v51] = v31;
                LODWORD(v71) = v71 + 1;
              }
              v31 = *(_QWORD *)(v30 + 16);
LABEL_51:
              while ( 1 )
              {
                v30 = *(_QWORD *)(v30 + 32);
                if ( !v30 )
                  goto LABEL_52;
                if ( (*(_BYTE *)(v30 + 3) & 0x10) == 0 )
                {
                  v44 = *(_QWORD *)(v30 + 16);
                  if ( v31 != v44 )
                  {
                    v31 = *(_QWORD *)(v30 + 16);
                    if ( **(_WORD **)(v44 + 16) == 12 )
                      goto LABEL_69;
                  }
                }
              }
            }
LABEL_98:
            v61 = a4;
            sub_1DC6D40(a4, v45);
            sub_1FD4240(v61, &v68, &v69);
            a4 = v61;
            v48 = (__int64)v69;
            v57 = v68;
            v56 = *(_DWORD *)(v61 + 16) + 1;
            goto LABEL_93;
          }
        }
        else
        {
          ++*(_QWORD *)a4;
        }
        v45 *= 2;
        goto LABEL_98;
      }
      while ( 1 )
      {
        v30 = *(_QWORD *)(v30 + 32);
        if ( !v30 )
          break;
        if ( (*(_BYTE *)(v30 + 3) & 0x10) == 0 )
          goto LABEL_48;
      }
    }
LABEL_52:
    sub_1DD5BC0(*(_QWORD *)(a1[5] + 784LL) + 16LL, (__int64)a2);
    v32 = (unsigned __int64 *)*((_QWORD *)a2 + 1);
    v33 = *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL;
    *v32 = v33 | *v32 & 7;
    *(_QWORD *)(v33 + 8) = v32;
    *(_QWORD *)a2 &= 7uLL;
    *((_QWORD *)a2 + 1) = 0;
    sub_1DD6E10(*(_QWORD *)(a1[5] + 784LL), v21, (__int64)a2);
    if ( v21 != (__int64 *)(*(_QWORD *)(a1[5] + 784LL) + 24LL) )
    {
      v34 = v21[8];
      v69 = (unsigned __int8 *)v34;
      if ( v34 )
      {
        v35 = (__int64)(a2 + 64);
        sub_1623A60((__int64)&v69, v34, 2);
        v36 = *((_QWORD *)a2 + 8);
        if ( !v36 )
          goto LABEL_56;
        goto LABEL_55;
      }
      v36 = *((_QWORD *)a2 + 8);
      v35 = (__int64)(a2 + 64);
      if ( v36 )
      {
LABEL_55:
        sub_161E7C0(v35, v36);
LABEL_56:
        v37 = v69;
        *((_QWORD *)a2 + 8) = v69;
        if ( v37 )
          sub_1623210((__int64)&v69, v37, v35);
      }
    }
    v38 = v70;
    v39 = (_QWORD **)&v70[(unsigned int)v71];
    if ( v39 != v70 )
    {
      v40 = (_QWORD **)v70;
      do
      {
        v41 = *v40++;
        sub_1DD5BC0(*(_QWORD *)(a1[5] + 784LL) + 16LL, (__int64)v41);
        v42 = (unsigned __int64 *)v41[1];
        v43 = *v41 & 0xFFFFFFFFFFFFFFF8LL;
        *v42 = v43 | *v42 & 7;
        *(_QWORD *)(v43 + 8) = v42;
        *v41 &= 7uLL;
        v41[1] = 0;
        sub_1DD6E10(*(_QWORD *)(a1[5] + 784LL), v21, (__int64)v41);
      }
      while ( v39 != v40 );
      v38 = v70;
    }
    if ( v38 != v72 )
      _libc_free((unsigned __int64)v38);
    return;
  }
LABEL_7:
  v17 = a1[7];
  if ( (int)v7 < 0 )
    v18 = *(_QWORD *)(*(_QWORD *)(v17 + 24) + 16 * (v7 & 0x7FFFFFFF) + 8);
  else
    v18 = *(_QWORD *)(*(_QWORD *)(v17 + 272) + 8LL * (unsigned int)v7);
  while ( v18 )
  {
    if ( (*(_BYTE *)(v18 + 3) & 0x10) == 0 && (*(_BYTE *)(v18 + 4) & 8) == 0 )
    {
      v63 = 0;
      goto LABEL_20;
    }
    v18 = *(_QWORD *)(v18 + 32);
  }
  if ( (unsigned __int8 *)a1[19] == a2 )
  {
    if ( a2 == *(unsigned __int8 **)(*((_QWORD *)a2 + 3) + 32LL) )
      v58 = 0;
    else
      v58 = *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL;
    a1[19] = v58;
  }
  v69 = a2;
  v65 = a4;
  if ( (unsigned __int8)sub_1FD4240(a4, (__int64 *)&v69, &v70) )
  {
    *v70 = -16;
    --*(_DWORD *)(v65 + 16);
    ++*(_DWORD *)(v65 + 20);
  }
  sub_1E16240((__int64)a2);
}
