// Function: sub_39BD220
// Address: 0x39bd220
//
void __fastcall sub_39BD220(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v5; // ecx
  _QWORD *v6; // rsi
  __int64 *v7; // rax
  __int64 v8; // rdx
  __int64 *v9; // r15
  __int64 v10; // r14
  __int64 *v11; // r9
  __int64 *v12; // r10
  __int64 v13; // rcx
  void *v14; // rdx
  char *v15; // rax
  unsigned __int64 v16; // r13
  char *v17; // r8
  char *v18; // rcx
  __int64 v19; // rsi
  __int64 *v20; // rax
  __int64 v21; // rax
  __int64 *v22; // rdx
  __int64 *v23; // rax
  __int64 v24; // rdx
  unsigned __int64 *v25; // r15
  __int64 v26; // rdx
  unsigned __int64 v27; // rsi
  unsigned __int64 v28; // rax
  int v29; // edx
  _QWORD *v30; // rsi
  _QWORD *v31; // r14
  __int64 v32; // rax
  _QWORD *v33; // r12
  __int64 v34; // r15
  __int64 v35; // rdi
  __int64 v36; // rax
  _BYTE *v37; // rsi
  __int64 v38; // rdx
  _QWORD *v39; // rax
  char **v40; // rax
  char **v41; // rbx
  char *v42; // r14
  char *v43; // r15
  __int64 v44; // rcx
  void *v45; // rdx
  char *v46; // rax
  unsigned __int64 v47; // r12
  unsigned __int64 *v48; // r14
  unsigned __int64 *v49; // r13
  __int64 *v50; // rdx
  char *v51; // rdx
  char *v52; // rax
  char *v53; // rcx
  __int64 *v54; // [rsp+0h] [rbp-90h]
  __int64 *v56; // [rsp+10h] [rbp-80h]
  __int64 *v57; // [rsp+18h] [rbp-78h]
  char **v58; // [rsp+18h] [rbp-78h]
  __int64 v59; // [rsp+20h] [rbp-70h]
  __int64 v60; // [rsp+20h] [rbp-70h]
  void *v61; // [rsp+28h] [rbp-68h]
  void *v62; // [rsp+28h] [rbp-68h]
  char *v63; // [rsp+28h] [rbp-68h]
  _QWORD v64[2]; // [rsp+30h] [rbp-60h] BYREF
  _QWORD v65[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v66; // [rsp+50h] [rbp-40h]

  v64[1] = a4;
  v5 = *(_DWORD *)(a1 + 112);
  v64[0] = a3;
  if ( v5 )
  {
    v6 = *(_QWORD **)(a1 + 104);
    if ( *v6 && *v6 != -8 )
    {
      v9 = *(__int64 **)(a1 + 104);
    }
    else
    {
      v7 = v6 + 1;
      do
      {
        do
        {
          v8 = *v7;
          v9 = v7++;
        }
        while ( !v8 );
      }
      while ( v8 == -8 );
    }
    v54 = &v6[v5];
    while ( v9 != v54 )
    {
      while ( 1 )
      {
        v10 = *v9;
        v11 = *(__int64 **)(*v9 + 32);
        v12 = *(__int64 **)(*v9 + 24);
        v13 = v11 - v12;
        if ( (char *)v11 - (char *)v12 <= 0 )
        {
LABEL_68:
          v16 = 0;
          sub_39BBE80(v12, v11);
        }
        else
        {
          v14 = &unk_435FF63;
          while ( 1 )
          {
            v56 = v12;
            v57 = v11;
            v59 = v13;
            v61 = v14;
            v15 = (char *)sub_2207800(8 * v13);
            v14 = v61;
            v11 = v57;
            v12 = v56;
            v16 = (unsigned __int64)v15;
            if ( v15 )
              break;
            v13 = v59 >> 1;
            if ( !(v59 >> 1) )
              goto LABEL_68;
          }
          sub_39BCCD0(v56, v57, v15, v59);
        }
        j_j___libc_free_0(v16);
        v17 = *(char **)(v10 + 32);
        v18 = *(char **)(v10 + 24);
        if ( v18 != v17 )
        {
          while ( 1 )
          {
            v20 = (__int64 *)v18;
            v18 += 8;
            if ( v17 == v18 )
              break;
            v19 = *((_QWORD *)v18 - 1);
            if ( v19 == v20[1] )
            {
              if ( v17 != (char *)v20 )
              {
                v50 = v20 + 2;
                if ( v17 == (char *)(v20 + 2) )
                  goto LABEL_63;
                while ( 1 )
                {
                  if ( v19 != *v50 )
                  {
                    v20[1] = *v50;
                    ++v20;
                  }
                  if ( v17 == (char *)++v50 )
                    break;
                  v19 = *v20;
                }
                v18 = (char *)(v20 + 1);
                if ( v20 + 1 != (__int64 *)v17 )
                {
LABEL_63:
                  v51 = *(char **)(v10 + 32);
                  if ( v17 != v51 )
                  {
                    v63 = v17;
                    v52 = (char *)memmove(v18, v17, v51 - v17);
                    v51 = *(char **)(v10 + 32);
                    v17 = v63;
                    v18 = v52;
                  }
                  v53 = &v18[v51 - v17];
                  if ( v53 != v51 )
                    *(_QWORD *)(v10 + 32) = v53;
                }
              }
              break;
            }
          }
        }
        v21 = v9[1];
        v22 = v9 + 1;
        if ( v21 == -8 || !v21 )
          break;
        ++v9;
        if ( v22 == v54 )
          goto LABEL_22;
      }
      v23 = v9 + 2;
      do
      {
        do
        {
          v24 = *v23;
          v9 = v23++;
        }
        while ( v24 == -8 );
      }
      while ( !v24 );
    }
  }
LABEL_22:
  sub_39BC0D0(a1);
  v25 = *(unsigned __int64 **)(a1 + 184);
  v26 = *(_QWORD *)(a1 + 176);
  v27 = *(unsigned int *)(a1 + 144);
  v28 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v25 - v26) >> 3);
  if ( v27 > v28 )
  {
    sub_39BC330(a1 + 176, v27 - v28);
  }
  else if ( v27 < v28 )
  {
    v48 = (unsigned __int64 *)(v26 + 24 * v27);
    if ( v25 != v48 )
    {
      v49 = (unsigned __int64 *)(v26 + 24 * v27);
      do
      {
        if ( *v49 )
          j_j___libc_free_0(*v49);
        v49 += 3;
      }
      while ( v25 != v49 );
      *(_QWORD *)(a1 + 184) = v48;
    }
  }
  v29 = *(_DWORD *)(a1 + 112);
  if ( v29 )
  {
    v30 = *(_QWORD **)(a1 + 104);
    v31 = v30;
    if ( *v30 == -8 || !*v30 )
    {
      do
      {
        do
        {
          v32 = v31[1];
          ++v31;
        }
        while ( !v32 );
      }
      while ( v32 == -8 );
    }
    v33 = &v30[v29];
    if ( v33 != v31 )
    {
      while ( 1 )
      {
        v34 = *v31;
        v35 = *(_QWORD *)(a1 + 176) + 24LL * (unsigned int)(*(_DWORD *)(*v31 + 16LL) % *(_DWORD *)(a1 + 144));
        v36 = *v31 + 8LL;
        v65[0] = v36;
        v37 = *(_BYTE **)(v35 + 8);
        if ( v37 == *(_BYTE **)(v35 + 16) )
        {
          sub_39BC570(v35, v37, v65);
        }
        else
        {
          if ( v37 )
          {
            *(_QWORD *)v37 = v36;
            v37 = *(_BYTE **)(v35 + 8);
          }
          *(_QWORD *)(v35 + 8) = v37 + 8;
        }
        v66 = 261;
        v65[0] = v64;
        *(_QWORD *)(v34 + 48) = sub_396F530(a2, (__int64)v65);
        v38 = v31[1];
        v39 = v31 + 1;
        if ( !v38 )
          break;
LABEL_36:
        if ( v38 == -8 )
          goto LABEL_35;
        if ( v39 == v33 )
          goto LABEL_40;
        v31 = v39;
      }
      do
      {
LABEL_35:
        v38 = v39[1];
        ++v39;
      }
      while ( !v38 );
      goto LABEL_36;
    }
  }
LABEL_40:
  v40 = *(char ***)(a1 + 184);
  v41 = *(char ***)(a1 + 176);
  v58 = v40;
  while ( v58 != v41 )
  {
    v42 = v41[1];
    v43 = *v41;
    v44 = (v42 - *v41) >> 3;
    if ( v42 - *v41 <= 0 )
    {
LABEL_48:
      v47 = 0;
      sub_39BC050(v43, v42);
    }
    else
    {
      v45 = &unk_435FF63;
      while ( 1 )
      {
        v60 = v44;
        v62 = v45;
        v46 = (char *)sub_2207800(8 * v44);
        v45 = v62;
        v47 = (unsigned __int64)v46;
        if ( v46 )
          break;
        v44 = v60 >> 1;
        if ( !(v60 >> 1) )
          goto LABEL_48;
      }
      sub_39BD150(v43, v42, v46, (char *)v60);
    }
    v41 += 3;
    j_j___libc_free_0(v47);
  }
}
