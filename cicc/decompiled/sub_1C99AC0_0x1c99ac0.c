// Function: sub_1C99AC0
// Address: 0x1c99ac0
//
void __fastcall sub_1C99AC0(unsigned __int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r12
  unsigned __int64 v7; // rbx
  _QWORD *v8; // rax
  __int64 v9; // rdi
  _QWORD *v10; // rsi
  __int64 v11; // rdx
  unsigned __int8 v12; // al
  _QWORD *v13; // rax
  __int64 v14; // r14
  _QWORD *v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // rdx
  _QWORD *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r8
  __int64 v21; // r15
  _BOOL4 v22; // ebx
  __int64 v23; // rax
  unsigned __int64 v24; // rdi
  __int64 *v25; // rbx
  __int64 *v26; // r15
  __int64 v27; // r9
  unsigned __int8 v28; // al
  _QWORD *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // r12
  _BOOL4 v32; // r15d
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // r9
  __int64 v36; // r14
  unsigned __int8 v37; // al
  __int64 v38; // rdx
  _QWORD *v39; // r10
  __int64 v40; // rsi
  int v41; // r9d
  char v42; // al
  int v43; // r9d
  unsigned int v44; // eax
  __int64 v45; // r15
  unsigned __int64 v46; // rax
  __int64 v47; // rcx
  unsigned __int64 v48; // r9
  _QWORD *v49; // rax
  _QWORD *v50; // rsi
  _QWORD *v51; // rax
  __int64 v52; // rdx
  _BOOL4 v53; // r9d
  __int64 v54; // rax
  __int64 v55; // [rsp+8h] [rbp-98h]
  __int64 v56; // [rsp+10h] [rbp-90h]
  _BOOL4 v57; // [rsp+10h] [rbp-90h]
  __int64 v58; // [rsp+18h] [rbp-88h]
  unsigned __int64 v59; // [rsp+18h] [rbp-88h]
  __int64 v60; // [rsp+18h] [rbp-88h]
  __int64 v61; // [rsp+20h] [rbp-80h]
  __int64 v62; // [rsp+20h] [rbp-80h]
  __int64 v63; // [rsp+20h] [rbp-80h]
  __int64 v64; // [rsp+20h] [rbp-80h]
  __int64 v65; // [rsp+20h] [rbp-80h]
  __int64 v66; // [rsp+20h] [rbp-80h]
  unsigned __int64 v67[2]; // [rsp+28h] [rbp-78h] BYREF
  __int64 v68; // [rsp+38h] [rbp-68h] BYREF
  _QWORD *v69; // [rsp+40h] [rbp-60h] BYREF
  __int64 v70; // [rsp+48h] [rbp-58h]
  _QWORD v71[10]; // [rsp+50h] [rbp-50h] BYREF

  v6 = a4;
  v7 = a1;
  v8 = *(_QWORD **)(a4 + 16);
  v67[0] = a1;
  if ( v8 )
  {
    v9 = a4 + 8;
    v10 = (_QWORD *)(a4 + 8);
    do
    {
      while ( 1 )
      {
        a4 = v8[2];
        v11 = v8[3];
        if ( v8[4] >= v7 )
          break;
        v8 = (_QWORD *)v8[3];
        if ( !v11 )
          goto LABEL_6;
      }
      v10 = v8;
      v8 = (_QWORD *)v8[2];
    }
    while ( a4 );
LABEL_6:
    if ( (_QWORD *)v9 != v10 && v10[4] <= v7 )
    {
      v29 = sub_1819210(a2, v67);
      v31 = v30;
      if ( v30 )
      {
        v32 = 1;
        if ( !v29 && v30 != a2 + 8 )
          v32 = *(_QWORD *)(v30 + 32) > v7;
        v33 = sub_22077B0(40);
        *(_QWORD *)(v33 + 32) = v67[0];
        sub_220F040(v32, v33, v31, a2 + 8);
        ++*(_QWORD *)(a2 + 40);
      }
      return;
    }
  }
  v12 = *(_BYTE *)(v7 + 16);
  if ( v12 <= 0x17u )
    return;
  if ( (v12 & 0xFD) != 0x4D )
  {
    if ( v12 == 71 )
    {
      sub_1C99AC0(*(_QWORD *)(v7 - 24), a2, a3, v6);
      return;
    }
    if ( v12 != 56 )
    {
      if ( v12 == 86 )
      {
        v71[0] = v7;
        v35 = 1;
        v69 = v71;
        v70 = 0x400000001LL;
        v36 = *(_QWORD *)(v7 - 24);
        while ( 1 )
        {
          v37 = *(_BYTE *)(v36 + 16);
          if ( v37 != 86 )
            break;
          if ( (unsigned int)v35 >= HIDWORD(v70) )
          {
            v64 = a3;
            sub_16CD150((__int64)&v69, v71, 0, 8, a3, v35);
            v35 = (unsigned int)v70;
            a3 = v64;
          }
          v69[v35] = v36;
          v35 = (unsigned int)(v70 + 1);
          LODWORD(v70) = v70 + 1;
          v36 = *(_QWORD *)(v36 - 24);
          if ( !v36 )
            BUG();
        }
        v38 = (unsigned int)v35;
        if ( !(_DWORD)v35 )
          goto LABEL_60;
        v39 = v69;
        while ( 1 )
        {
          if ( v37 <= 0x17u )
            goto LABEL_52;
          v40 = v39[v38 - 1];
          if ( v37 == 54 )
            break;
          if ( v37 != 87 || **(_QWORD **)(v40 - 24) != *(_QWORD *)v36 )
            goto LABEL_52;
          v65 = a3;
          v42 = sub_1C957E0(v36, v40, v38, a4, a3);
          a3 = v65;
          if ( v42 )
          {
            LODWORD(v35) = v43 - 1;
            LODWORD(v70) = v35;
            v36 = *(_QWORD *)(v36 - 24);
          }
          else
          {
            v36 = *(_QWORD *)(v36 - 48);
            LODWORD(v35) = v70;
          }
          v38 = (unsigned int)v35;
          if ( !(_DWORD)v35 )
            goto LABEL_60;
          v37 = *(_BYTE *)(v36 + 16);
        }
        if ( **(_QWORD **)(v40 - 24) != *(_QWORD *)v36 )
        {
LABEL_52:
          v41 = v70;
          goto LABEL_53;
        }
        v41 = v35 - 1;
        LODWORD(v70) = v41;
LABEL_53:
        if ( v41 )
        {
          if ( v39 != v71 )
            _libc_free((unsigned __int64)v39);
        }
        else
        {
LABEL_60:
          sub_1C99AC0(v36, a2, a3, v6);
          if ( v69 != v71 )
            _libc_free((unsigned __int64)v69);
        }
        return;
      }
      if ( v12 != 78 )
        return;
      v34 = *(_QWORD *)(v7 - 24);
      if ( *(_BYTE *)(v34 + 16) || (*(_BYTE *)(v34 + 33) & 0x20) == 0 || *(_DWORD *)(v34 + 36) != 3660 )
        return;
    }
    sub_1C99AC0(*(_QWORD *)(v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF)), a2, a3, v6);
    return;
  }
  v13 = *(_QWORD **)(a3 + 16);
  v14 = a3 + 8;
  if ( !v13 )
    goto LABEL_17;
  v15 = (_QWORD *)(a3 + 8);
  do
  {
    while ( 1 )
    {
      v16 = v13[2];
      v17 = v13[3];
      if ( v13[4] >= v7 )
        break;
      v13 = (_QWORD *)v13[3];
      if ( !v17 )
        goto LABEL_15;
    }
    v15 = v13;
    v13 = (_QWORD *)v13[2];
  }
  while ( v16 );
LABEL_15:
  if ( (_QWORD *)v14 == v15 || v15[4] > v7 )
  {
LABEL_17:
    v61 = a3;
    v18 = sub_1819210(a3, v67);
    v20 = v61;
    v21 = v19;
    if ( v19 )
    {
      v22 = v18 || v14 == v19 || v7 < *(_QWORD *)(v19 + 32);
      v23 = sub_22077B0(40);
      *(_QWORD *)(v23 + 32) = v67[0];
      sub_220F040(v22, v23, v21, v14);
      v20 = v61;
      v7 = v67[0];
      ++*(_QWORD *)(v61 + 40);
    }
    v69 = 0;
    v70 = 0;
    v71[0] = 0;
    if ( *(_BYTE *)(v7 + 16) == 77 )
    {
      v44 = *(_DWORD *)(v7 + 20) & 0xFFFFFFF;
      if ( !v44 )
        return;
      v45 = 0;
      do
      {
        if ( (*(_BYTE *)(v7 + 23) & 0x40) != 0 )
          v46 = *(_QWORD *)(v7 - 8);
        else
          v46 = v7 - 24LL * v44;
        v47 = 3 * v45;
        v66 = v20;
        ++v45;
        v68 = *(_QWORD *)(v46 + 8 * v47);
        sub_15E88C0((__int64)&v69, &v68);
        v20 = v66;
        v44 = *(_DWORD *)(v7 + 20) & 0xFFFFFFF;
      }
      while ( v44 > (unsigned int)v45 );
    }
    else
    {
      v62 = v20;
      v68 = *(_QWORD *)(v7 - 48);
      sub_15E88C0((__int64)&v69, &v68);
      v68 = *(_QWORD *)(v7 - 24);
      sub_15E88C0((__int64)&v69, &v68);
      v20 = v62;
    }
    v24 = (unsigned __int64)v69;
    v25 = (__int64 *)v70;
    v63 = a2 + 8;
    v26 = v69;
    if ( v69 != (_QWORD *)v70 )
    {
      while ( 1 )
      {
        v27 = *v26;
        v68 = v27;
        v28 = *(_BYTE *)(v27 + 16);
        if ( v28 <= 0x17u || (v28 & 0xFD) != 0x4D )
        {
          if ( sub_1C96F00(v27) )
            goto LABEL_28;
          v49 = *(_QWORD **)(a2 + 16);
          if ( v49 )
          {
            v50 = (_QWORD *)(a2 + 8);
            do
            {
              if ( v49[4] < v48 )
              {
                v49 = (_QWORD *)v49[3];
              }
              else
              {
                v50 = v49;
                v49 = (_QWORD *)v49[2];
              }
            }
            while ( v49 );
            if ( (_QWORD *)v63 != v50 && v50[4] <= v48 )
              goto LABEL_28;
          }
          v56 = v20;
          v59 = v48;
          v51 = sub_1819210(a2, (unsigned __int64 *)&v68);
          v27 = v59;
          v20 = v56;
          if ( v52 )
          {
            v53 = v51 || v63 == v52 || v59 < *(_QWORD *)(v52 + 32);
            v55 = v56;
            v57 = v53;
            v60 = v52;
            v54 = sub_22077B0(40);
            *(_QWORD *)(v54 + 32) = v68;
            sub_220F040(v57, v54, v60, v63);
            ++*(_QWORD *)(a2 + 40);
            v27 = v68;
            v20 = v55;
          }
        }
        v58 = v20;
        sub_1C99AC0(v27, a2, v20, v6);
        v20 = v58;
LABEL_28:
        if ( v25 == ++v26 )
        {
          v24 = (unsigned __int64)v69;
          break;
        }
      }
    }
    if ( v24 )
      j_j___libc_free_0(v24, v71[0] - v24);
  }
}
