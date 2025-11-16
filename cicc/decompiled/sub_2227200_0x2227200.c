// Function: sub_2227200
// Address: 0x2227200
//
__int64 __fastcall sub_2227200(__int64 a1, __int64 a2, char a3, __int64 a4, unsigned int a5, _QWORD *a6)
{
  __int64 v6; // r15
  __int64 v9; // rbp
  __int64 v10; // r12
  __int64 *v11; // rax
  __int64 v12; // r14
  _DWORD *v13; // r12
  __int64 v14; // rax
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // r15
  __int64 result; // rax
  int *v18; // rsi
  unsigned __int64 v19; // rcx
  __int64 v20; // rdx
  unsigned __int64 v21; // r13
  __int64 v22; // rbx
  int v23; // eax
  char *v24; // rax
  unsigned __int64 v25; // r13
  unsigned __int64 *v26; // rbx
  unsigned __int64 v27; // r12
  __int64 v28; // rax
  __int64 v29; // kr00_8
  unsigned __int64 v30; // r13
  unsigned __int64 v31; // rdx
  unsigned __int64 v32; // rsi
  unsigned __int64 v33; // r13
  unsigned __int64 v34; // rdx
  unsigned __int64 v35; // rsi
  unsigned __int64 v36; // rcx
  _QWORD *v37; // rsi
  unsigned __int64 v38; // rdx
  unsigned __int64 v39; // rbp
  unsigned __int64 v40; // rax
  _DWORD *v41; // rsi
  __int64 v42; // r14
  __int64 *v45; // [rsp+18h] [rbp-F0h]
  unsigned __int64 v46; // [rsp+20h] [rbp-E8h]
  int *v49; // [rsp+38h] [rbp-D0h]
  int v50; // [rsp+50h] [rbp-B8h]
  int v51; // [rsp+50h] [rbp-B8h]
  bool v52; // [rsp+57h] [rbp-B1h]
  unsigned __int64 v53; // [rsp+58h] [rbp-B0h]
  int v54; // [rsp+60h] [rbp-A8h]
  unsigned __int64 v55; // [rsp+60h] [rbp-A8h]
  unsigned __int64 v56; // [rsp+68h] [rbp-A0h]
  int v57; // [rsp+8Ch] [rbp-7Ch] BYREF
  char *v58; // [rsp+90h] [rbp-78h] BYREF
  unsigned __int64 v59; // [rsp+98h] [rbp-70h]
  _QWORD v60[2]; // [rsp+A0h] [rbp-68h] BYREF
  _QWORD *v61; // [rsp+B0h] [rbp-58h] BYREF
  unsigned __int64 v62; // [rsp+B8h] [rbp-50h]
  _QWORD v63[9]; // [rsp+C0h] [rbp-48h] BYREF

  v6 = a4 + 208;
  v9 = sub_2243120(a4 + 208);
  v10 = sub_22091A0(&qword_4FD6908);
  v11 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a4 + 208) + 24LL) + 8 * v10);
  v12 = *v11;
  v45 = v11;
  if ( !*v11 )
  {
    v42 = sub_22077B0(0xA0u);
    *(_DWORD *)(v42 + 8) = 0;
    *(_QWORD *)(v42 + 16) = 0;
    *(_QWORD *)(v42 + 24) = 0;
    *(_QWORD *)v42 = off_4A048C0;
    *(_BYTE *)(v42 + 32) = 0;
    *(_QWORD *)(v42 + 36) = 0;
    *(_QWORD *)(v42 + 48) = 0;
    *(_QWORD *)(v42 + 56) = 0;
    *(_QWORD *)(v42 + 64) = 0;
    *(_QWORD *)(v42 + 72) = 0;
    *(_QWORD *)(v42 + 80) = 0;
    *(_QWORD *)(v42 + 88) = 0;
    *(_QWORD *)(v42 + 96) = 0;
    *(_DWORD *)(v42 + 104) = 0;
    *(_BYTE *)(v42 + 152) = 0;
    sub_22443D0(v42, v6);
    sub_2209690(*(_QWORD *)(a4 + 208), (volatile signed __int32 *)v42, v10);
    v12 = *v45;
  }
  v13 = (_DWORD *)*a6;
  v14 = a6[1];
  if ( *(_DWORD *)*a6 == *(_DWORD *)(v12 + 108) )
  {
    v18 = *(int **)(v12 + 80);
    v19 = *(_QWORD *)(v12 + 88);
    v57 = *(_DWORD *)(v12 + 104);
    v49 = v18;
    if ( v14 )
      ++v13;
    v46 = v19;
  }
  else
  {
    v49 = *(int **)(v12 + 64);
    v15 = *(_QWORD *)(v12 + 72);
    v57 = *(_DWORD *)(v12 + 100);
    v46 = v15;
  }
  v16 = ((*(__int64 (__fastcall **)(__int64, __int64, _DWORD *, _DWORD *))(*(_QWORD *)v9 + 40LL))(
           v9,
           2048,
           v13,
           &v13[v14])
       - (__int64)v13) >> 2;
  if ( v16 )
  {
    v59 = 0;
    v58 = (char *)v60;
    LODWORD(v60[0]) = 0;
    sub_22519B0(&v58, 2 * v16);
    v20 = *(int *)(v12 + 96);
    v21 = v59;
    v22 = v16 - v20;
    v23 = *(_DWORD *)(v12 + 96);
    if ( (__int64)(v16 - v20) > 0 )
    {
      if ( (int)v20 < 0 )
        v22 = v16;
      if ( *(_QWORD *)(v12 + 24) )
      {
        sub_2251AD0(&v58, 0, v59, 2 * v22, 0);
        v24 = (char *)sub_2244D30(
                        v58,
                        *(unsigned int *)(v12 + 40),
                        *(_QWORD *)(v12 + 16),
                        *(_QWORD *)(v12 + 24),
                        v13,
                        &v13[v22]);
        v21 = (v24 - v58) >> 2;
        if ( v21 > v59 )
          sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::erase");
        v59 = (v24 - v58) >> 2;
        *(_DWORD *)v24 = 0;
        v23 = *(_DWORD *)(v12 + 96);
      }
      else
      {
        sub_2251C20(&v58, 0, v59, v13, v22);
        v23 = *(_DWORD *)(v12 + 96);
        v21 = v59;
      }
    }
    if ( v23 > 0 )
    {
      v38 = 3;
      v39 = v21 + 1;
      v51 = *(_DWORD *)(v12 + 36);
      v40 = (unsigned __int64)v58;
      if ( v58 != (char *)v60 )
        v38 = v60[0];
      if ( v39 > v38 )
      {
        sub_2251880(&v58, v21, 0, 0, 1);
        v40 = (unsigned __int64)v58;
      }
      *(_DWORD *)(v40 + 4 * v21) = v51;
      v59 = v21 + 1;
      *(_DWORD *)(v40 + 4 * v21 + 4) = 0;
      if ( v22 < 0 )
      {
        sub_2251AD0(&v58, v21 + 1, 0, -v22, *(unsigned int *)(v12 + 112));
        if ( v16 > 0xFFFFFFFFFFFFFFFLL - v59 )
          sub_4262D8((__int64)"basic_string::append");
        v41 = v13;
      }
      else
      {
        v41 = &v13[v22];
        if ( *(int *)(v12 + 96) > 0xFFFFFFFFFFFFFFFLL - v39 )
          sub_4262D8((__int64)"basic_string::append");
      }
      sub_2251F20(&v58, v41);
      v21 = v59;
    }
    v25 = v46 + v21;
    v50 = *(_DWORD *)(a4 + 24) & 0xB0;
    if ( (*(_DWORD *)(a4 + 24) & 0x200) != 0 )
      v25 += *(_QWORD *)(v12 + 56);
    v62 = 0;
    v61 = v63;
    LODWORD(v63[0]) = 0;
    sub_22519B0(&v61, 2 * v25);
    v26 = (unsigned __int64 *)&v57;
    v27 = *(_QWORD *)(a4 + 16);
    v28 = v27 - v25;
    v52 = v27 > v25 && v50 == 16;
    v53 = v27 - v25;
    do
    {
      v29 = v28;
      v28 = *(unsigned __int8 *)v26;
      switch ( *(_BYTE *)v26 )
      {
        case 0:
          if ( v52 )
            goto LABEL_37;
          break;
        case 1:
          v33 = v62;
          if ( v52 )
          {
LABEL_37:
            v28 = sub_2251AD0(&v61, v62, 0, v53, a5);
          }
          else
          {
            v28 = (__int64)v61;
            v34 = 3;
            if ( v61 != v63 )
              v34 = v63[0];
            v55 = v62 + 1;
            if ( v62 + 1 > v34 )
            {
              sub_2251880(&v61, v62, 0, 0, 1);
              v28 = (__int64)v61;
            }
            *(_DWORD *)(v28 + 4 * v33) = a5;
            v62 = v55;
            *(_DWORD *)(v28 + 4 * v33 + 4) = 0;
          }
          break;
        case 2:
          v28 = a4;
          if ( (*(_BYTE *)(a4 + 25) & 2) != 0 )
          {
            v32 = *(_QWORD *)(v12 + 48);
            if ( *(_QWORD *)(v12 + 56) > 0xFFFFFFFFFFFFFFFLL - v62 )
              sub_4262D8((__int64)"basic_string::append");
            goto LABEL_39;
          }
          return result;
        case 3:
          if ( v46 )
          {
            v30 = v62;
            v31 = 3;
            v56 = v62 + 1;
            v54 = *v49;
            v28 = (__int64)v61;
            if ( v61 != v63 )
              v31 = v63[0];
            if ( v62 + 1 > v31 )
            {
              sub_2251880(&v61, v62, 0, 0, 1);
              v28 = (__int64)v61;
            }
            *(_DWORD *)(v28 + 4 * v30) = v54;
            v62 = v56;
            *(_DWORD *)(v28 + 4 * v30 + 4) = 0;
          }
          break;
        case 4:
          v32 = (unsigned __int64)v58;
LABEL_39:
          v28 = sub_2251F20(&v61, v32);
          break;
        default:
          v28 = v29;
          break;
      }
      v26 = (unsigned __int64 *)((char *)v26 + 1);
    }
    while ( &v58 != (char **)v26 );
    if ( v46 > 1 )
    {
      if ( v46 - 1 > 0xFFFFFFFFFFFFFFFLL - v62 )
        sub_4262D8((__int64)"basic_string::append");
      sub_2251F20(&v61, v49 + 1);
      v35 = v62;
      if ( v27 > v62 )
        goto LABEL_43;
    }
    else
    {
      v35 = v62;
      if ( v27 > v62 )
      {
LABEL_43:
        v36 = v27 - v35;
        if ( v50 != 32 )
          v35 = 0;
        sub_2251AD0(&v61, v35, 0, v36, a5);
        goto LABEL_46;
      }
    }
    LODWORD(v27) = v35;
LABEL_46:
    v37 = v61;
    if ( !a3 )
    {
      (*(void (__fastcall **)(__int64, _QWORD *, _QWORD))(*(_QWORD *)a2 + 96LL))(a2, v61, (int)v27);
      v37 = v61;
    }
    if ( v37 != v63 )
      j___libc_free_0((unsigned __int64)v37);
    if ( v58 != (char *)v60 )
      j___libc_free_0((unsigned __int64)v58);
  }
  *(_QWORD *)(a4 + 16) = 0;
  return a2;
}
