// Function: sub_1E07D70
// Address: 0x1e07d70
//
__int64 __fastcall sub_1E07D70(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  _QWORD *v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rax
  __int64 v7; // rcx
  int v8; // r8d
  int v9; // r9d
  __int64 v10; // rdx
  __int64 v11; // rcx
  int v12; // r8d
  int v13; // r9d
  __int64 v14; // rax
  int v15; // eax
  __int64 v16; // rdx
  __int64 *v17; // rax
  int v18; // r9d
  __int64 v19; // r8
  __int64 v20; // rcx
  _BYTE *v21; // rsi
  __int64 *v22; // r13
  __int64 v23; // rdx
  __int64 v24; // rcx
  int v25; // r8d
  int v26; // r9d
  __int64 v27; // rdx
  int v28; // r8d
  int v29; // r9d
  __int64 v30; // rcx
  int v31; // r8d
  int v32; // r9d
  __int64 v33; // rdx
  char *v34; // rbx
  __int64 v35; // rax
  __int64 *v36; // r12
  __int64 *v37; // rax
  __int64 v38; // rdi
  __int64 *v39; // rbx
  _QWORD *v40; // rbx
  _QWORD *v41; // r12
  unsigned __int64 v42; // rdi
  __int64 result; // rax
  _QWORD *v44; // rbx
  _QWORD *v45; // r12
  unsigned __int64 v46; // rdi
  unsigned int v49; // [rsp+4Ch] [rbp-314h]
  __int64 *v50; // [rsp+50h] [rbp-310h]
  __int64 *v51; // [rsp+58h] [rbp-308h]
  __int64 v52; // [rsp+68h] [rbp-2F8h] BYREF
  __int64 v53; // [rsp+70h] [rbp-2F0h] BYREF
  __int64 v54; // [rsp+78h] [rbp-2E8h] BYREF
  __int64 v55; // [rsp+80h] [rbp-2E0h] BYREF
  __int64 v56; // [rsp+88h] [rbp-2D8h] BYREF
  _QWORD *v57; // [rsp+90h] [rbp-2D0h] BYREF
  _BYTE *v58; // [rsp+98h] [rbp-2C8h]
  _BYTE *v59; // [rsp+A0h] [rbp-2C0h]
  __int64 v60; // [rsp+A8h] [rbp-2B8h] BYREF
  _QWORD *v61; // [rsp+B0h] [rbp-2B0h]
  __int64 v62; // [rsp+B8h] [rbp-2A8h]
  unsigned int v63; // [rsp+C0h] [rbp-2A0h]
  __int64 v64; // [rsp+C8h] [rbp-298h]
  __int64 *v65; // [rsp+D0h] [rbp-290h] BYREF
  int v66; // [rsp+D8h] [rbp-288h]
  char v67; // [rsp+E0h] [rbp-280h] BYREF
  char *v68; // [rsp+120h] [rbp-240h] BYREF
  __int64 v69; // [rsp+128h] [rbp-238h]
  _QWORD v70[70]; // [rsp+130h] [rbp-230h] BYREF

  v2 = *(_QWORD *)(a1 + 64);
  sub_1E05DB0(a1 + 24);
  *(_QWORD *)(a1 + 64) = v2;
  *(_DWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_BYTE *)(a1 + 72) = 0;
  *(_DWORD *)(a1 + 76) = 0;
  v58 = 0;
  v59 = 0;
  v3 = (_QWORD *)sub_22077B0(8);
  v68 = (char *)v70;
  v4 = (__int64)(v3 + 1);
  v57 = v3;
  *v3 = 0;
  v69 = 0x100000000LL;
  v5 = *(_QWORD *)(a1 + 64);
  v59 = (_BYTE *)v4;
  v6 = *(__int64 **)(v5 + 328);
  v58 = (_BYTE *)v4;
  v60 = 0;
  v65 = v6;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  sub_1E05890((__int64)&v68, &v65, v4, v7, v8, v9);
  sub_1E04970(a1, &v68, v10, v11, v12, v13);
  if ( v68 != (char *)v70 )
    _libc_free((unsigned __int64)v68);
  v14 = **(_QWORD **)a1;
  v68 = (char *)v70;
  v69 = 0x4000000001LL;
  v52 = v14;
  v70[0] = v14;
  v56 = v14;
  if ( (unsigned __int8)sub_1E060E0((__int64)&v60, &v56, &v65) )
    *((_DWORD *)sub_1E071E0((__int64)&v60, &v52) + 3) = 0;
  v15 = v69;
  v49 = 0;
  if ( (_DWORD)v69 )
  {
    while ( 1 )
    {
      v16 = *(_QWORD *)&v68[8 * v15 - 8];
      LODWORD(v69) = v15 - 1;
      v53 = v16;
      v17 = sub_1E071E0((__int64)&v60, &v53);
      if ( *((_DWORD *)v17 + 2) )
        goto LABEL_7;
      ++v49;
      v19 = v53;
      v20 = v49;
      v17[3] = v53;
      *((_DWORD *)v17 + 2) = v49;
      v21 = v58;
      *((_DWORD *)v17 + 4) = v49;
      if ( v21 == v59 )
      {
        sub_1D4AF10((__int64)&v57, v21, &v53);
        v19 = v53;
      }
      else
      {
        if ( v21 )
        {
          *(_QWORD *)v21 = v19;
          v21 = v58;
          v19 = v53;
        }
        v58 = v21 + 8;
      }
      sub_1E06CD0((__int64)&v65, v19, v64, v20, v19, v18);
      v50 = &v65[v66];
      if ( v65 != v50 )
      {
        v22 = v65;
        while ( 1 )
        {
          while ( 1 )
          {
            v54 = *v22;
            v55 = v54;
            if ( (unsigned __int8)sub_1E060E0((__int64)&v60, &v55, &v56) )
            {
              if ( (_QWORD *)v56 != &v61[9 * v63] )
              {
                v33 = *(unsigned int *)(v56 + 8);
                if ( (_DWORD)v33 )
                  break;
              }
            }
            v51 = sub_1E071E0((__int64)&v60, &v54);
            sub_1E05890((__int64)&v68, &v54, v23, v24, v25, v26);
            *((_DWORD *)v51 + 3) = v49;
            sub_1E05890((__int64)(v51 + 5), &v53, v27, v49, v28, v29);
LABEL_16:
            if ( v50 == ++v22 )
              goto LABEL_22;
          }
          if ( v54 == v53 )
            goto LABEL_16;
          ++v22;
          sub_1E05890(v56 + 40, &v53, v33, v30, v31, v32);
          if ( v50 == v22 )
          {
LABEL_22:
            v50 = v65;
            break;
          }
        }
      }
      if ( v50 == (__int64 *)&v67 )
      {
LABEL_7:
        v15 = v69;
        if ( !(_DWORD)v69 )
          break;
      }
      else
      {
        _libc_free((unsigned __int64)v50);
        v15 = v69;
        if ( !(_DWORD)v69 )
          break;
      }
    }
  }
  if ( v68 != (char *)v70 )
    _libc_free((unsigned __int64)v68);
  sub_1E07970((__int64 *)&v57, a1, 0);
  if ( a2 )
    *(_BYTE *)(a2 + 144) = 1;
  if ( *(_DWORD *)(a1 + 8) )
  {
    v34 = **(char ***)a1;
    v68 = v34;
    v35 = sub_22077B0(56);
    v36 = (__int64 *)v35;
    if ( v35 )
    {
      *(_QWORD *)v35 = v34;
      *(_QWORD *)(v35 + 8) = 0;
      *(_DWORD *)(v35 + 16) = 0;
      *(_QWORD *)(v35 + 24) = 0;
      *(_QWORD *)(v35 + 32) = 0;
      *(_QWORD *)(v35 + 40) = 0;
      *(_QWORD *)(v35 + 48) = -1;
    }
    v37 = sub_1E063B0(a1 + 24, (__int64 *)&v68);
    v38 = v37[1];
    v39 = v37;
    v37[1] = (__int64)v36;
    if ( v38 )
    {
      sub_1E046F0(v38);
      v36 = (__int64 *)v39[1];
    }
    *(_QWORD *)(a1 + 56) = v36;
    sub_1E07680((__int64)&v57, a1, v36);
    if ( v63 )
    {
      v40 = v61;
      v41 = &v61[9 * v63];
      do
      {
        if ( *v40 != -16 && *v40 != -8 )
        {
          v42 = v40[5];
          if ( (_QWORD *)v42 != v40 + 7 )
            _libc_free(v42);
        }
        v40 += 9;
      }
      while ( v41 != v40 );
    }
  }
  else if ( v63 )
  {
    v44 = v61;
    v45 = &v61[9 * v63];
    do
    {
      if ( *v44 != -16 && *v44 != -8 )
      {
        v46 = v44[5];
        if ( (_QWORD *)v46 != v44 + 7 )
          _libc_free(v46);
      }
      v44 += 9;
    }
    while ( v45 != v44 );
  }
  result = j___libc_free_0(v61);
  if ( v57 )
    return j_j___libc_free_0(v57, v59 - (_BYTE *)v57);
  return result;
}
