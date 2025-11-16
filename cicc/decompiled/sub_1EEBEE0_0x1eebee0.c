// Function: sub_1EEBEE0
// Address: 0x1eebee0
//
__int64 __fastcall sub_1EEBEE0(__int64 a1, __int64 a2)
{
  __int64 (*v2)(); // rax
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // rbx
  unsigned int v7; // eax
  __int64 *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // r13
  __int64 *v13; // rdi
  __int64 v14; // rax
  __int64 (*v15)(void); // rdx
  __int64 (*v16)(); // rax
  __int64 v17; // rax
  unsigned int *v18; // r12
  unsigned __int16 *v19; // rax
  int v20; // edi
  unsigned __int64 v21; // r9
  int v22; // ecx
  unsigned __int64 i; // r8
  __int64 v24; // r14
  unsigned int v25; // esi
  __int64 *v26; // r12
  __int64 **v27; // rcx
  __int64 **v28; // rax
  __int64 v29; // rdx
  int v30; // edx
  unsigned __int16 *v31; // rdi
  int v32; // eax
  unsigned __int16 *v33; // rdi
  int v34; // ecx
  unsigned __int16 *v35; // rax
  int v36; // edx
  __int64 v37; // rbx
  unsigned int v38; // r15d
  __int64 v39; // r12
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // r10
  unsigned int v43; // eax
  __int16 v44; // di
  _WORD *v45; // rdx
  unsigned __int16 *v46; // rax
  unsigned __int16 v47; // di
  unsigned __int16 v48; // dx
  unsigned __int16 *v49; // r11
  unsigned __int16 *v50; // rsi
  unsigned __int16 *v51; // rax
  unsigned __int16 v52; // cx
  __int64 v53; // rax
  __int16 *v54; // r11
  __int16 v55; // ax
  unsigned __int16 *v56; // rax
  unsigned __int16 *v58; // r10
  int v59; // eax
  unsigned __int16 *v60; // r10
  unsigned __int64 v61; // rcx
  unsigned __int16 *v62; // rdx
  int v63; // eax
  __int64 v64; // rdx
  __int64 v65; // rdx
  int v66; // eax
  unsigned __int16 v67; // cx
  _QWORD *v68; // [rsp+0h] [rbp-A0h]
  __int64 v69; // [rsp+8h] [rbp-98h]
  __int64 v70; // [rsp+10h] [rbp-90h]
  __int64 v71; // [rsp+18h] [rbp-88h]
  unsigned int *v72; // [rsp+20h] [rbp-80h]
  unsigned int v73; // [rsp+2Ch] [rbp-74h]
  unsigned __int16 v74; // [rsp+2Ch] [rbp-74h]
  unsigned __int64 v75; // [rsp+30h] [rbp-70h] BYREF
  __int64 v76; // [rsp+38h] [rbp-68h]
  __int64 v77; // [rsp+40h] [rbp-60h]
  unsigned __int64 v78[2]; // [rsp+50h] [rbp-50h] BYREF
  int v79; // [rsp+60h] [rbp-40h]

  v2 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 112LL);
  if ( v2 == sub_1D00B10 )
  {
    v75 = 0;
    v76 = 0;
    v77 = 0;
    BUG();
  }
  v68 = *(_QWORD **)(a2 + 40);
  v4 = v2();
  v77 = 0;
  v5 = *(_QWORD *)(a2 + 8);
  v6 = v4;
  LODWORD(v4) = *(_DWORD *)(v4 + 16);
  v75 = 0;
  v76 = 0;
  LODWORD(v78[0]) = -1;
  v7 = (unsigned int)(v4 + 31) >> 5;
  if ( v7 )
    sub_1CFD340((__int64)&v75, 0, v7, v78);
  v8 = *(__int64 **)(a1 + 8);
  v9 = *v8;
  v10 = v8[1];
  if ( v9 == v10 )
LABEL_82:
    BUG();
  while ( *(_UNKNOWN **)v9 != &unk_4FCFA48 )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_82;
  }
  v71 = *(_QWORD *)a2;
  v11 = v5;
  v12 = 0;
  v69 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4FCFA48);
  sub_210D7F0(v69, v11);
  v13 = *(__int64 **)(a2 + 16);
  v78[0] = 0;
  v78[1] = 0;
  v79 = 0;
  v14 = *v13;
  v15 = *(__int64 (**)(void))(*v13 + 48);
  if ( v15 != sub_1D90020 )
  {
    v12 = v15();
    v14 = **(_QWORD **)(a2 + 16);
  }
  v16 = *(__int64 (**)())(v14 + 112);
  if ( v16 == sub_1D00B10 )
  {
    v79 = 0;
    (*(void (__fastcall **)(__int64, __int64, unsigned __int64 *, _QWORD))(*(_QWORD *)v12 + 192LL))(v12, a2, v78, 0);
    BUG();
  }
  v17 = v16();
  v79 = 0;
  v18 = (unsigned int *)v17;
  (*(void (__fastcall **)(__int64, __int64, unsigned __int64 *, _QWORD))(*(_QWORD *)v12 + 192LL))(v12, a2, v78, 0);
  v19 = (unsigned __int16 *)(*(__int64 (__fastcall **)(unsigned int *, __int64))(*(_QWORD *)v18 + 24LL))(v18, a2);
  v20 = 0;
  v21 = 0;
  v22 = *v19;
  for ( i = (unsigned __int64)v19; (_WORD)v22; ++v20 )
  {
    if ( (*(_QWORD *)(v78[0] + 8 * ((unsigned __int64)(unsigned __int16)v22 >> 6)) & (1LL << v22)) != 0 )
    {
      v58 = (unsigned __int16 *)(*((_QWORD *)v18 + 7)
                               + 2LL * *(unsigned int *)(*((_QWORD *)v18 + 1) + 24LL * (unsigned __int16)v22 + 4));
      v59 = *v58;
      v60 = v58 + 1;
      v61 = (unsigned int)(v59 + v22);
      if ( !(_WORD)v59 )
        v60 = 0;
LABEL_61:
      v62 = v60;
      while ( v62 )
      {
        ++v62;
        *(_QWORD *)(v78[0] + ((v61 >> 3) & 0x1FF8)) |= 1LL << v61;
        v63 = *(v62 - 1);
        v60 = 0;
        v61 = (unsigned int)(v63 + v61);
        if ( !(_WORD)v63 )
          goto LABEL_61;
      }
    }
    v22 = *(unsigned __int16 *)(i + 2LL * (unsigned int)(v20 + 1));
  }
  v73 = v18[4];
  if ( v73 <= 1 )
    goto LABEL_30;
  v72 = v18;
  v24 = 24;
  v25 = 1;
  v70 = v6;
  do
  {
    while ( 1 )
    {
      v21 = v78[0];
      v26 = (__int64 *)(v78[0] + 8LL * (v25 >> 6));
      if ( (*v26 & (1LL << v25)) == 0 )
      {
        v27 = (__int64 **)*((_QWORD *)v72 + 33);
        v28 = (__int64 **)*((_QWORD *)v72 + 32);
        if ( v27 != v28 )
          break;
      }
LABEL_14:
      ++v25;
      v24 += 24;
      if ( v73 == v25 )
        goto LABEL_29;
    }
    i = v25 >> 3;
    while ( 1 )
    {
      if ( *((_BYTE *)*v28 + 30) )
      {
        v29 = **v28;
        if ( (unsigned int)i < *(unsigned __int16 *)(v29 + 22) )
        {
          v30 = *(unsigned __int8 *)(*(_QWORD *)(v29 + 8) + (v25 >> 3));
          if ( _bittest(&v30, v25 & 7) )
            break;
        }
      }
      if ( v27 == ++v28 )
        goto LABEL_14;
    }
    v31 = (unsigned __int16 *)(*((_QWORD *)v72 + 7) + 2LL * *(unsigned int *)(*((_QWORD *)v72 + 1) + v24 + 4));
    v32 = *v31;
    v33 = v31 + 1;
    v34 = v32 + v25;
    if ( !(_WORD)v32 )
      v33 = 0;
LABEL_24:
    v35 = v33;
    if ( v33 )
    {
      while ( (*(_QWORD *)(v78[0] + 8 * ((unsigned __int64)(unsigned __int16)v34 >> 6)) & (1LL << v34)) != 0 )
      {
        v36 = *v35;
        v33 = 0;
        ++v35;
        i = (unsigned int)(v36 + v34);
        if ( !(_WORD)v36 )
          goto LABEL_24;
        v34 += v36;
        if ( !v35 )
          goto LABEL_28;
      }
      goto LABEL_14;
    }
LABEL_28:
    v37 = *v26 | (1LL << v25++);
    v24 += 24;
    *v26 = v37;
  }
  while ( v73 != v25 );
LABEL_29:
  v6 = v70;
LABEL_30:
  v38 = *(_DWORD *)(v6 + 16);
  if ( v38 > 1 )
  {
    v39 = 8;
    LODWORD(i) = 1;
    while ( 1 )
    {
      v40 = (unsigned int)i >> 6;
      if ( (*(_QWORD *)(v78[0] + 8 * v40) & (1LL << i)) != 0 )
        goto LABEL_50;
      if ( (i & 0x80000000) != 0LL )
        v41 = *(_QWORD *)(v68[3] + 16 * (i & 0x7FFFFFFF) + 8);
      else
        v41 = *(_QWORD *)(v68[34] + v39);
      if ( v41 )
      {
        if ( (*(_BYTE *)(v41 + 3) & 0x10) != 0
          || (v64 = *(_QWORD *)(v41 + 32)) != 0 && (*(_BYTE *)(v64 + 3) & 0x10) != 0 )
        {
          v42 = *(_QWORD *)(v6 + 8);
          v21 = *(_QWORD *)(v6 + 56);
          v43 = *(_DWORD *)(v42 + 3 * v39 + 16);
          v44 = i * (v43 & 0xF);
          v45 = (_WORD *)(v21 + 2LL * (v43 >> 4));
          v46 = v45 + 1;
          v47 = *v45 + v44;
          v74 = 0;
          v48 = 0;
LABEL_38:
          v49 = v46;
          while ( 1 )
          {
            v50 = v49;
            if ( !v49 )
            {
              v52 = v74;
              v53 = 0;
              goto LABEL_42;
            }
            v51 = (unsigned __int16 *)(*(_QWORD *)(v6 + 48) + 4LL * v47);
            v52 = *v51;
            v48 = v51[1];
            if ( *v51 )
              break;
LABEL_75:
            v67 = *v49;
            v46 = 0;
            ++v49;
            if ( !v67 )
              goto LABEL_38;
            v47 += v67;
          }
          while ( 1 )
          {
            v53 = v21 + 2LL * *(unsigned int *)(v42 + 24LL * v52 + 8);
            if ( v53 )
              break;
            if ( !v48 )
            {
              v74 = v52;
              goto LABEL_75;
            }
            v52 = v48;
            v48 = 0;
          }
LABEL_42:
          v54 = (__int16 *)v53;
          while ( v50 )
          {
            while ( 1 )
            {
              v21 = v52 >> 6;
              if ( (*(_QWORD *)(v78[0] + 8 * v21) & (1LL << v52)) == 0 )
              {
                v21 = v75;
                *(_DWORD *)(v75 + 4LL * (v52 >> 5)) &= ~(1 << v52);
              }
              v55 = *v54++;
              v52 += v55;
              if ( v55 )
                break;
              if ( v48 )
              {
                v53 = *(_QWORD *)(v6 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v6 + 8) + 24LL * v48 + 8);
                v52 = v48;
                v48 = 0;
                goto LABEL_42;
              }
              v48 = *v50;
              v47 += *v50;
              if ( !*v50 )
              {
                v53 = 0;
                v50 = 0;
                goto LABEL_42;
              }
              ++v50;
              v56 = (unsigned __int16 *)(*(_QWORD *)(v6 + 48) + 4LL * v47);
              v52 = *v56;
              v48 = v56[1];
              v21 = *(unsigned int *)(*(_QWORD *)(v6 + 8) + 24LL * *v56 + 8);
              v54 = (__int16 *)(*(_QWORD *)(v6 + 56) + 2 * v21);
              if ( !v50 )
                goto LABEL_50;
            }
          }
          goto LABEL_50;
        }
      }
      if ( (*(_QWORD *)(v68[35] + 8 * v40) & (1LL << i)) == 0 )
      {
LABEL_50:
        i = (unsigned int)(i + 1);
        v39 += 8;
        if ( (_DWORD)i == v38 )
          break;
      }
      else
      {
        v65 = (unsigned int)i >> 5;
        v66 = 1 << i;
        i = (unsigned int)(i + 1);
        v39 += 8;
        *(_DWORD *)(v75 + 4 * v65) &= ~v66;
        if ( (_DWORD)i == v38 )
          break;
      }
    }
  }
  if ( (*(_BYTE *)(v71 + 32) & 0xFu) - 7 <= 1 && !(unsigned __int8)sub_15E3650(v71, 0) )
    sub_1560180(v71 + 112, 27);
  sub_210DB00(v69, v71, v75, (__int64)(v76 - v75) >> 2, i, v21);
  _libc_free(v78[0]);
  if ( v75 )
    j_j___libc_free_0(v75, v77 - v75);
  return 0;
}
