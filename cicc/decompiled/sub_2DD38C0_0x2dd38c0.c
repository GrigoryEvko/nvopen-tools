// Function: sub_2DD38C0
// Address: 0x2dd38c0
//
void __fastcall sub_2DD38C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r13d
  __int64 v7; // rbx
  char *v8; // r14
  __int64 v9; // r12
  __int64 v10; // rdi
  unsigned int v11; // r12d
  __int64 v12; // r14
  __int64 *v13; // r15
  int v14; // r13d
  __int64 v15; // rdi
  __int64 v16; // rdx
  char **v17; // r14
  __int64 v18; // rcx
  __int64 v19; // r12
  unsigned __int64 v20; // rbx
  __int64 v21; // r13
  char *v22; // rdi
  char *v23; // rdx
  __int64 v24; // r13
  char *v25; // r12
  __int64 v26; // r14
  char *v27; // r15
  int v28; // ebx
  __int64 v29; // rdi
  __int64 v30; // rdx
  __int64 *v31; // r15
  __int64 *v32; // rbx
  int v33; // r12d
  __int64 v34; // rdi
  __int64 v35; // rdi
  __int64 v36; // rbx
  char *v37; // rdi
  __int64 v39; // [rsp+20h] [rbp-90h]
  unsigned int v40; // [rsp+28h] [rbp-88h]
  int v41; // [rsp+2Ch] [rbp-84h]
  unsigned int v42; // [rsp+2Ch] [rbp-84h]
  char *v43; // [rsp+30h] [rbp-80h] BYREF
  __int64 v44; // [rsp+38h] [rbp-78h]
  _BYTE v45[48]; // [rsp+40h] [rbp-70h] BYREF
  int v46; // [rsp+70h] [rbp-40h]
  int v47; // [rsp+78h] [rbp-38h]

  if ( a1 != a2 && a1 + 80 != a2 )
  {
    v39 = a1 + 80;
    do
    {
      v6 = 0;
      v7 = *(unsigned int *)(v39 + 8);
      v8 = *(char **)v39;
      v9 = *(_QWORD *)v39 + 8 * v7;
      v40 = *(_DWORD *)(v39 + 8);
      if ( *(_QWORD *)v39 != v9 )
      {
        do
        {
          v10 = *(_QWORD *)v8;
          v8 += 8;
          v6 += sub_39FAC40(v10);
        }
        while ( (char *)v9 != v8 );
      }
      v41 = *(_DWORD *)(v39 + 72);
      v11 = v41 * v6;
      v12 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
      if ( *(_QWORD *)a1 == v12 )
        goto LABEL_18;
      v13 = *(__int64 **)a1;
      v14 = 0;
      do
      {
        v15 = *v13++;
        v14 += sub_39FAC40(v15);
      }
      while ( (__int64 *)v12 != v13 );
      if ( v11 >= *(_DWORD *)(a1 + 72) * v14 )
      {
LABEL_18:
        v23 = v45;
        v44 = 0x600000000LL;
        v43 = v45;
        if ( v40 )
        {
          sub_2DD3500((__int64)&v43, (char **)v39, (__int64)v45, a4, a5, a6);
          v23 = v43;
          v7 = (unsigned int)v44;
          v41 = *(_DWORD *)(v39 + 72);
        }
        v24 = v39;
        v25 = &v23[8 * v7];
        v26 = v39;
        v46 = *(_DWORD *)(v39 + 64);
        v47 = v41;
        if ( v25 != v23 )
        {
LABEL_21:
          v27 = v23;
          v28 = 0;
          do
          {
            v29 = *(_QWORD *)v27;
            v27 += 8;
            v28 += sub_39FAC40(v29);
          }
          while ( v25 != v27 );
          v42 = v41 * v28;
          goto LABEL_24;
        }
        while ( 1 )
        {
          v42 = 0;
LABEL_24:
          v30 = *(_QWORD *)(v24 - 80);
          v31 = (__int64 *)(v30 + 8LL * *(unsigned int *)(v24 - 72));
          if ( (__int64 *)v30 == v31 )
            break;
          v32 = *(__int64 **)(v24 - 80);
          v33 = 0;
          do
          {
            v34 = *v32++;
            v33 += sub_39FAC40(v34);
          }
          while ( v31 != v32 );
          v24 -= 80;
          if ( *(_DWORD *)(v26 - 8) * v33 <= v42 )
            break;
          v35 = v26;
          v26 = v24;
          sub_2DD3500(v35, (char **)v24, v30, a4, a5, a6);
          v23 = v43;
          v36 = (unsigned int)v44;
          *(_DWORD *)(v24 + 144) = *(_DWORD *)(v24 + 64);
          v25 = &v23[8 * v36];
          *(_DWORD *)(v24 + 152) = *(_DWORD *)(v24 + 72);
          v41 = v47;
          if ( v25 != v23 )
            goto LABEL_21;
        }
        sub_2DD3500(v26, &v43, v30, a4, a5, a6);
        v37 = v43;
        *(_DWORD *)(v26 + 64) = v46;
        *(_DWORD *)(v26 + 72) = v47;
        if ( v37 != v45 )
          _libc_free((unsigned __int64)v37);
        v19 = v39 + 80;
      }
      else
      {
        v16 = v40;
        v43 = v45;
        v44 = 0x600000000LL;
        if ( v40 )
        {
          sub_2DD3500((__int64)&v43, (char **)v39, v40, a4, a5, a6);
          v41 = *(_DWORD *)(v39 + 72);
        }
        v17 = (char **)v39;
        v18 = 0xCCCCCCCCCCCCCCCDLL;
        v19 = v39 + 80;
        v46 = *(_DWORD *)(v39 + 64);
        v47 = v41;
        v20 = 0xCCCCCCCCCCCCCCCDLL * ((v39 - a1) >> 4);
        if ( v39 - a1 > 0 )
        {
          do
          {
            v21 = (__int64)v17;
            v17 -= 10;
            sub_2DD3500(v21, v17, v16, v18, a5, a6);
            *(_DWORD *)(v21 + 64) = *(_DWORD *)(v21 - 16);
            *(_DWORD *)(v21 + 72) = *(_DWORD *)(v21 - 8);
            --v20;
          }
          while ( v20 );
        }
        sub_2DD3500(a1, &v43, v16, v18, a5, a6);
        v22 = v43;
        *(_DWORD *)(a1 + 64) = v46;
        *(_DWORD *)(a1 + 72) = v47;
        if ( v22 != v45 )
          _libc_free((unsigned __int64)v22);
      }
      v39 = v19;
    }
    while ( a2 != v19 );
  }
}
