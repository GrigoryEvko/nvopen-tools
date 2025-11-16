// Function: sub_22F7250
// Address: 0x22f7250
//
__int64 *__fastcall sub_22F7250(__int64 *a1, __int64 a2, __int64 a3, unsigned int *a4)
{
  unsigned int *v6; // rbx
  char *v7; // rax
  unsigned int v8; // r13d
  size_t v9; // rdx
  _BYTE *v10; // rax
  int v11; // r14d
  int v12; // edx
  int v13; // r12d
  __int64 v14; // rbx
  unsigned __int64 v16; // rax
  __int64 v17; // r8
  __int64 v18; // r9
  size_t v19; // rcx
  size_t v20; // rsi
  char *v21; // rdx
  __int64 *v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 v25; // rsi
  _DWORD *v26; // rdi
  unsigned int *v27; // rax
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rcx
  unsigned int *v31; // r13
  unsigned int v32; // ebx
  __int64 v33; // rax
  size_t v34; // r8
  size_t v35; // rax
  size_t v36; // rcx
  __int64 v37; // rbx
  __int64 v38; // r13
  size_t v39; // rdx
  __int64 v40; // rax
  size_t v41; // rcx
  const char *v42; // r13
  size_t v43; // rax
  unsigned int v44; // eax
  __int64 v45; // r14
  int v46; // eax
  unsigned int v47; // r15d
  int v48; // r8d
  int v49; // r14d
  int v50; // edx
  int v51; // r12d
  unsigned int v52; // r14d
  int v53; // eax
  int v54; // r15d
  int v55; // r13d
  int v56; // edx
  int v57; // r12d
  __int64 v58; // rax
  __int64 v59; // [rsp-10h] [rbp-230h]
  __int64 v60; // [rsp+8h] [rbp-218h]
  __int64 v61; // [rsp+10h] [rbp-210h]
  unsigned int v63; // [rsp+24h] [rbp-1FCh]
  unsigned int *v64; // [rsp+30h] [rbp-1F0h]
  __int64 v65; // [rsp+38h] [rbp-1E8h]
  unsigned int v66; // [rsp+38h] [rbp-1E8h]
  int v67; // [rsp+38h] [rbp-1E8h]
  int v68; // [rsp+38h] [rbp-1E8h]
  __int64 v69; // [rsp+48h] [rbp-1D8h] BYREF
  char *v70; // [rsp+50h] [rbp-1D0h] BYREF
  size_t v71; // [rsp+58h] [rbp-1C8h]
  _QWORD v72[2]; // [rsp+60h] [rbp-1C0h] BYREF
  __int64 v73[6]; // [rsp+70h] [rbp-1B0h] BYREF
  __int64 v74[2]; // [rsp+A0h] [rbp-180h] BYREF
  char *v75; // [rsp+B0h] [rbp-170h]
  size_t v76; // [rsp+B8h] [rbp-168h]
  __int16 v77; // [rsp+C0h] [rbp-160h]
  _BYTE *v78; // [rsp+D0h] [rbp-150h] BYREF
  __int64 v79; // [rsp+D8h] [rbp-148h]
  __int64 v80; // [rsp+E0h] [rbp-140h]
  _BYTE v81[312]; // [rsp+E8h] [rbp-138h] BYREF

  v6 = a4;
  v7 = *(char **)(*(_QWORD *)(a3 + 184) + 8LL * *a4);
  v8 = *a4;
  v9 = 0;
  v60 = (__int64)v7;
  v70 = v7;
  if ( v7 )
    v9 = strlen(v7);
  v10 = *(_BYTE **)(a2 + 80);
  v71 = v9;
  v78 = v10;
  v79 = *(unsigned int *)(a2 + 88);
  if ( (unsigned __int8)sub_22F50A0((__int64 *)&v78, v70, v9) )
  {
    *v6 = v8 + 1;
    v11 = sub_22F59B0(a2, *(_DWORD *)(a2 + 64));
    v13 = v12;
    v14 = sub_22077B0(0x58u);
    if ( v14 )
      sub_314D360(v14, v11, v13, (_DWORD)v70, v71, v8, v60, 0);
LABEL_6:
    *a1 = v14;
    return a1;
  }
  v65 = *(_QWORD *)(a2 + 32) + 80LL * *(_QWORD *)(a2 + 40);
  v16 = sub_C935B0(&v70, *(unsigned __int8 **)(a2 + 144), *(_QWORD *)(a2 + 152), 0);
  v19 = v71;
  v20 = 0;
  if ( v16 < v71 )
  {
    v20 = v71 - v16;
    v19 = v16;
  }
  v21 = &v70[v19];
  v22 = *(__int64 **)(a2 + 8);
  v23 = *(_QWORD *)(a2 + 24);
  v72[0] = v21;
  v24 = *(unsigned int *)(a2 + 72);
  v72[1] = v20;
  v25 = *(_QWORD *)(a2 + 16);
  v73[4] = v23;
  v73[3] = v25;
  v26 = (_DWORD *)(*(_QWORD *)(a2 + 32) + 80 * v24);
  v73[2] = (__int64)v22;
  v27 = sub_22F5220(v26, v65, (__int64)v72, v23, v17, v18, v22, v25);
  v30 = *v6;
  v31 = v27;
  v63 = *v6;
  if ( (unsigned int *)v65 == v27 )
    goto LABEL_30;
  v64 = v6;
  v61 = 0;
  do
  {
    v32 = sub_22F5520(*(void ***)(a2 + 8), *(_QWORD *)(a2 + 16), v31, v70, v71, *(_BYTE *)(a2 + 48));
    if ( v32 )
    {
      sub_22F3E40(&v78, (__int64)v31, a2);
      sub_22F4690(
        v74,
        (__int64 *)&v78,
        (__int64 (__fastcall ***)(_QWORD))a3,
        *(_QWORD *)(*(_QWORD *)(a3 + 184) + 8LL * *v64),
        (char *)v32,
        0,
        v64);
      if ( v74[0] )
      {
        *a1 = v74[0];
        return a1;
      }
      if ( v32 == 2 )
      {
        v33 = v61;
        if ( v78[44] == 3 )
          v33 = (__int64)v31;
        v61 = v33;
      }
      if ( *v64 != v63 )
      {
        *a1 = 0;
        return a1;
      }
    }
    v31 += 20;
  }
  while ( (unsigned int *)v65 != v31 );
  v6 = v64;
  if ( !v61 )
    goto LABEL_30;
  sub_22F3E40(v73, v61, a2);
  if ( v70[2] == 61 )
    goto LABEL_43;
  v34 = v71;
  if ( v71 >= 2 )
    v34 = 2;
  sub_22F4690(&v69, v73, (__int64 (__fastcall ***)(_QWORD))a3, (int)v70, (char *)v34, 1, v64);
  v29 = v59;
  if ( !v69 )
  {
LABEL_30:
    if ( v70[1] != 45 )
    {
      v39 = v71;
      if ( v71 >= 2 )
        v39 = 2;
      v40 = (*(__int64 (__fastcall **)(__int64, char *, size_t, __int64, __int64, __int64))(*(_QWORD *)a3 + 16LL))(
              a3,
              v70,
              v39,
              v30,
              v28,
              v29);
      v41 = 0;
      v42 = (const char *)v40;
      v43 = v71;
      if ( v71 > 1 )
      {
        v41 = v71 - 2;
        v43 = 2;
      }
      v76 = v41;
      v75 = &v70[v43];
      v44 = *v6;
      v77 = 1288;
      v78 = v81;
      v66 = v44;
      LOBYTE(v74[0]) = 45;
      v79 = 0;
      v80 = 256;
      sub_CA0EC0((__int64)v74, (__int64)&v78);
      v45 = sub_22F3850(a3, v78, v79);
      if ( v78 != v81 )
        _libc_free((unsigned __int64)v78);
      *(_QWORD *)(*(_QWORD *)(a3 + 184) + 8LL * v66) = v45;
      v46 = sub_22F59B0(a2, *(_DWORD *)(a2 + 68));
      v47 = *v6;
      v48 = 0;
      v49 = v46;
      v51 = v50;
      if ( v42 )
        v48 = strlen(v42);
      v67 = v48;
      v14 = sub_22077B0(0x58u);
      if ( v14 )
        sub_314D360(v14, v49, v51, (_DWORD)v42, v67, v47, (__int64)v42, 0);
      goto LABEL_6;
    }
LABEL_43:
    v52 = (*v6)++;
    v53 = sub_22F59B0(a2, *(_DWORD *)(a2 + 68));
    v54 = v71;
    v55 = v53;
    v57 = v56;
    v68 = (int)v70;
    v58 = sub_22077B0(0x58u);
    v14 = v58;
    if ( v58 )
      sub_314D360(v58, v55, v57, v68, v54, v52, v60, 0);
    goto LABEL_6;
  }
  v35 = v71;
  v36 = 0;
  if ( v71 > 1 )
  {
    v36 = v71 - 2;
    v35 = 2;
  }
  v37 = *v64;
  v75 = &v70[v35];
  v76 = v36;
  v77 = 1288;
  LOBYTE(v74[0]) = 45;
  v78 = v81;
  v79 = 0;
  v80 = 256;
  sub_CA0EC0((__int64)v74, (__int64)&v78);
  v38 = sub_22F3850(a3, v78, v79);
  if ( v78 != v81 )
    _libc_free((unsigned __int64)v78);
  *(_QWORD *)(*(_QWORD *)(a3 + 184) + 8 * v37) = v38;
  *a1 = v69;
  return a1;
}
