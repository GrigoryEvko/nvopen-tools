// Function: sub_30F5D70
// Address: 0x30f5d70
//
bool __fastcall sub_30F5D70(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r14
  __int64 v3; // r13
  _QWORD **v4; // r15
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r9
  unsigned __int64 v9; // r12
  bool v10; // zf
  __int64 v11; // rsi
  __int64 **v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 **v15; // r14
  __int64 v16; // rbx
  unsigned int v17; // eax
  unsigned int v18; // r13d
  unsigned __int16 v19; // ax
  __int64 v20; // r8
  __int64 **v21; // rdi
  unsigned __int64 v22; // rsi
  char *v23; // r8
  unsigned __int64 v24; // rdx
  __int64 v25; // rdx
  _QWORD *v26; // r13
  _QWORD *v27; // rbx
  unsigned __int64 v28; // r12
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rdi
  __int64 v32; // rax
  __int64 v33; // rbx
  unsigned __int64 *v34; // r13
  __int64 v35; // rax
  unsigned __int64 v36; // rcx
  unsigned __int64 v37; // rsi
  int v38; // edx
  unsigned __int64 *v39; // rax
  unsigned __int64 v40; // rdi
  unsigned __int64 v41; // rdi
  char *v42; // r13
  __int64 v43; // [rsp+0h] [rbp-E0h]
  __int64 v44; // [rsp+18h] [rbp-C8h]
  __int64 v46; // [rsp+28h] [rbp-B8h]
  __int64 v47; // [rsp+30h] [rbp-B0h]
  unsigned int v48; // [rsp+3Ch] [rbp-A4h]
  __int64 v49; // [rsp+40h] [rbp-A0h]
  __int64 v50; // [rsp+48h] [rbp-98h]
  unsigned __int64 v51; // [rsp+58h] [rbp-88h] BYREF
  _QWORD *v52; // [rsp+60h] [rbp-80h] BYREF
  unsigned int v53; // [rsp+68h] [rbp-78h]
  int v54; // [rsp+6Ch] [rbp-74h]
  _QWORD v55[14]; // [rsp+70h] [rbp-70h] BYREF

  v48 = sub_DFB3D0(*(_QWORD **)(a1 + 256));
  v2 = sub_30F35F0(a1);
  v43 = v2[5];
  v44 = v2[4];
  if ( v43 == v44 )
    return *(_DWORD *)(a2 + 8) != 0;
  v3 = a1;
  v4 = (_QWORD **)v2;
  do
  {
    v5 = v3;
    v46 = *(_QWORD *)v44 + 48LL;
    v47 = *(_QWORD *)(*(_QWORD *)v44 + 56LL);
    if ( v47 == v46 )
      goto LABEL_35;
    do
    {
      if ( !v47 )
        BUG();
      if ( (unsigned __int8)(*(_BYTE *)(v47 - 24) - 61) <= 1u )
      {
        v6 = sub_22077B0(0x70u);
        v9 = v6;
        if ( v6 )
          sub_30F4EC0(v6, v47 - 24, *(_QWORD *)(v5 + 240), *(_QWORD *)(v5 + 248));
        v10 = *(_BYTE *)v9 == 0;
        v51 = v9;
        if ( v10 )
          goto LABEL_40;
        v11 = *(unsigned int *)(a2 + 8);
        v12 = *(__int64 ***)a2;
        LODWORD(v13) = *(_DWORD *)(a2 + 8);
        v49 = *(_QWORD *)a2 + 80 * v11;
        if ( v49 != *(_QWORD *)a2 )
        {
          v14 = v5;
          v15 = *(__int64 ***)a2;
          v16 = v14;
          while ( 1 )
          {
            v50 = **v15;
            LOWORD(v17) = sub_30F50C0(
                            v9,
                            v50,
                            *(_DWORD *)(v16 + 232),
                            v4,
                            *(_QWORD *)(v16 + 272),
                            *(_QWORD *)(v16 + 264));
            v18 = v17;
            v19 = sub_30F4FC0(v9, v50, v48, *(_QWORD *)(v16 + 264));
            v7 = v18;
            LOWORD(v7) = BYTE1(v18);
            if ( BYTE1(v18) )
            {
              if ( (_BYTE)v18 )
                break;
            }
            if ( HIBYTE(v19) && (_BYTE)v19 )
              break;
            v15 += 10;
            if ( (__int64 **)v49 == v15 )
            {
              v5 = v16;
              v11 = *(unsigned int *)(a2 + 8);
              v12 = *(__int64 ***)a2;
              LODWORD(v13) = *(_DWORD *)(a2 + 8);
              v21 = (__int64 **)(*(_QWORD *)a2 + 80 * v11);
              goto LABEL_17;
            }
          }
          v32 = v16;
          v33 = (__int64)v15;
          v34 = &v51;
          v5 = v32;
          v35 = *(unsigned int *)(v33 + 8);
          v36 = *(_QWORD *)v33;
          v37 = v35 + 1;
          v38 = v35;
          if ( v35 + 1 > (unsigned __int64)*(unsigned int *)(v33 + 12) )
          {
            if ( v36 > (unsigned __int64)&v51 || (unsigned __int64)&v51 >= v36 + 8 * v35 )
            {
              sub_30F56B0(v33, v37, v35, v36, v20, v8);
              v35 = *(unsigned int *)(v33 + 8);
              v36 = *(_QWORD *)v33;
              v38 = *(_DWORD *)(v33 + 8);
            }
            else
            {
              v42 = (char *)&v51 - v36;
              sub_30F56B0(v33, v37, v35, v36, v20, v8);
              v36 = *(_QWORD *)v33;
              v35 = *(unsigned int *)(v33 + 8);
              v34 = (unsigned __int64 *)&v42[*(_QWORD *)v33];
              v38 = *(_DWORD *)(v33 + 8);
            }
          }
          v39 = (unsigned __int64 *)(v36 + 8 * v35);
          if ( v39 )
          {
            *v39 = *v34;
            *v34 = 0;
            v9 = v51;
            ++*(_DWORD *)(v33 + 8);
            if ( !v9 )
              goto LABEL_33;
          }
          else
          {
            *(_DWORD *)(v33 + 8) = v38 + 1;
          }
LABEL_40:
          v40 = *(_QWORD *)(v9 + 64);
          if ( v40 != v9 + 80 )
            _libc_free(v40);
          v41 = *(_QWORD *)(v9 + 24);
          if ( v41 != v9 + 40 )
            _libc_free(v41);
          j_j___libc_free_0(v9);
          goto LABEL_33;
        }
        v21 = *(__int64 ***)a2;
LABEL_17:
        v22 = v11 + 1;
        v55[0] = v9;
        v54 = 8;
        v23 = (char *)&v52;
        v52 = v55;
        v51 = 0;
        v24 = *(unsigned int *)(a2 + 12);
        v53 = 1;
        if ( v22 > v24 )
        {
          if ( v12 > &v52 || v21 <= &v52 )
          {
            sub_30F5C00(a2, v22, v24, v7, (__int64)&v52, v8);
            v23 = (char *)&v52;
            LODWORD(v13) = *(_DWORD *)(a2 + 8);
            v21 = (__int64 **)(*(_QWORD *)a2 + 80LL * (unsigned int)v13);
          }
          else
          {
            sub_30F5C00(a2, v22, v24, v7, (__int64)&v52, v8);
            v23 = (char *)(*(_QWORD *)a2 + (char *)&v52 - (char *)v12);
            v13 = *(unsigned int *)(a2 + 8);
            v21 = (__int64 **)(*(_QWORD *)a2 + 80 * v13);
          }
        }
        if ( v21 )
        {
          *v21 = (__int64 *)(v21 + 2);
          v21[1] = (__int64 *)0x800000000LL;
          v25 = *((unsigned int *)v23 + 2);
          if ( (_DWORD)v25 )
            sub_30F57A0((__int64)v21, (__int64)v23, v25, v7, (__int64)v23, v8);
          LODWORD(v13) = *(_DWORD *)(a2 + 8);
        }
        v26 = v52;
        *(_DWORD *)(a2 + 8) = v13 + 1;
        v27 = &v26[v53];
        if ( v26 != v27 )
        {
          do
          {
            v28 = *--v27;
            if ( v28 )
            {
              v29 = *(_QWORD *)(v28 + 64);
              if ( v29 != v28 + 80 )
                _libc_free(v29);
              v30 = *(_QWORD *)(v28 + 24);
              if ( v30 != v28 + 40 )
                _libc_free(v30);
              j_j___libc_free_0(v28);
            }
          }
          while ( v26 != v27 );
          v27 = v52;
        }
        if ( v27 != v55 )
          _libc_free((unsigned __int64)v27);
      }
LABEL_33:
      v47 = *(_QWORD *)(v47 + 8);
    }
    while ( v46 != v47 );
    v3 = v5;
LABEL_35:
    v44 += 8;
  }
  while ( v43 != v44 );
  return *(_DWORD *)(a2 + 8) != 0;
}
