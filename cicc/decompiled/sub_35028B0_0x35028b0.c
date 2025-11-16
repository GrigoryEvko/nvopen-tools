// Function: sub_35028B0
// Address: 0x35028b0
//
__int64 __fastcall sub_35028B0(unsigned int *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 i; // rdx
  __int64 v13; // r15
  __int64 v14; // r13
  unsigned __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // r14
  unsigned int v18; // r13d
  __int64 result; // rax
  __int64 v20; // rbx
  __int64 v21; // r14
  __int64 v22; // r10
  unsigned __int64 v23; // r12
  char **v24; // r9
  int *v25; // rsi
  int v26; // edi
  int v27; // esi
  unsigned __int64 v28; // rdi
  __int64 v29; // r12
  char *v30; // rsi
  __int64 v31; // rdx
  _BYTE *v32; // rdi
  __int64 v33; // r12
  _QWORD *v34; // rax
  _QWORD *v35; // r9
  char *v36; // r12
  __int64 v37; // rax
  __int64 v38; // [rsp+8h] [rbp-C8h]
  _QWORD *v39; // [rsp+10h] [rbp-C0h]
  char **v40; // [rsp+18h] [rbp-B8h]
  char v41; // [rsp+18h] [rbp-B8h]
  _DWORD *v43; // [rsp+30h] [rbp-A0h] BYREF
  _BYTE *v44; // [rsp+38h] [rbp-98h]
  __int64 v45; // [rsp+40h] [rbp-90h]
  _BYTE v46[64]; // [rsp+48h] [rbp-88h] BYREF
  int v47; // [rsp+88h] [rbp-48h]
  __int64 v48; // [rsp+90h] [rbp-40h]

  *a1 = a2;
  v8 = a1[130];
  ++a1[1];
  v9 = (__int64)(*(_QWORD *)(a5 + 104) - *(_QWORD *)(a5 + 96)) >> 3;
  if ( (unsigned int)v9 != v8 )
  {
    if ( (unsigned int)v9 >= v8 )
    {
      if ( (unsigned int)v9 > (unsigned __int64)a1[131] )
      {
        sub_C8D5F0((__int64)(a1 + 128), a1 + 132, (unsigned int)v9, 0x18u, a5, a6);
        v8 = a1[130];
      }
      v10 = *((_QWORD *)a1 + 64);
      v11 = v10 + 24 * v8;
      for ( i = v10 + 24LL * (unsigned int)v9; i != v11; v11 += 24 )
      {
        if ( v11 )
        {
          *(_QWORD *)(v11 + 16) = 0;
          *(_OWORD *)v11 = 0;
          *(_QWORD *)(v11 + 8) = 0;
        }
      }
    }
    a1[130] = v9;
  }
  v13 = *((_QWORD *)a1 + 6);
  *((_QWORD *)a1 + 5) = 0;
  v38 = (__int64)(a1 + 12);
  v14 = v13 + 112LL * a1[14];
  while ( v13 != v14 )
  {
    while ( 1 )
    {
      v14 -= 112;
      v15 = *(_QWORD *)(v14 + 8);
      if ( v15 == v14 + 24 )
        break;
      _libc_free(v15);
      if ( v13 == v14 )
        goto LABEL_14;
    }
  }
LABEL_14:
  v16 = *a1;
  a1[14] = 0;
  LODWORD(v16) = *(_DWORD *)(*(_QWORD *)(a4 + 8) + 24 * v16 + 16);
  v17 = *(_QWORD *)(a4 + 56) + 2LL * ((unsigned int)v16 >> 12);
  v18 = v16 & 0xFFF;
  result = (__int64)a1;
  v20 = v17;
  v21 = result;
  do
  {
    if ( !v20 )
      break;
    v22 = *(unsigned int *)(v21 + 56);
    v23 = *(unsigned int *)(v21 + 60);
    v45 = 0x400000000LL;
    v48 = 0;
    v24 = (char **)&v43;
    v25 = (int *)(a3 + 216LL * v18);
    v26 = *v25;
    v44 = v46;
    v43 = v25 + 2;
    v27 = v22;
    v47 = v26;
    v28 = *(_QWORD *)(v21 + 48);
    if ( v22 + 1 > v23 )
    {
      if ( v28 > (unsigned __int64)&v43 || (unsigned __int64)&v43 >= v28 + 112 * v22 )
      {
        sub_3502770(v38, v22 + 1, (__int64)&v43, a3, a5, (__int64)&v43);
        v22 = *(unsigned int *)(v21 + 56);
        v28 = *(_QWORD *)(v21 + 48);
        v24 = (char **)&v43;
        v27 = *(_DWORD *)(v21 + 56);
      }
      else
      {
        v36 = (char *)&v43 - v28;
        sub_3502770(v38, v22 + 1, (__int64)&v43 - v28, a3, a5, (__int64)&v43);
        v28 = *(_QWORD *)(v21 + 48);
        v22 = *(unsigned int *)(v21 + 56);
        v24 = (char **)&v36[v28];
        v27 = *(_DWORD *)(v21 + 56);
      }
    }
    v29 = v28 + 112 * v22;
    if ( v29 )
    {
      v30 = *v24;
      *(_QWORD *)(v29 + 16) = 0x400000000LL;
      *(_QWORD *)v29 = v30;
      *(_QWORD *)(v29 + 8) = v29 + 24;
      v31 = *((unsigned int *)v24 + 4);
      if ( (_DWORD)v31 )
      {
        v40 = v24;
        sub_35018C0(v29 + 8, v24 + 1, v31, 0x400000000LL, a5, (__int64)v24);
        v24 = v40;
      }
      *(_DWORD *)(v29 + 88) = *((_DWORD *)v24 + 22);
      *(_QWORD *)(v29 + 96) = v24[12];
      *(_QWORD *)(v29 + 104) = v24[13];
      v27 = *(_DWORD *)(v21 + 56);
    }
    v32 = v44;
    *(_DWORD *)(v21 + 56) = v27 + 1;
    if ( v32 != v46 )
      _libc_free((unsigned __int64)v32);
    v33 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v21 + 32) + 424LL) + 8LL * v18);
    if ( !v33 )
    {
      v39 = *(_QWORD **)(v21 + 32);
      v41 = qword_501EA48[8];
      v34 = (_QWORD *)sub_22077B0(0x68u);
      v35 = v39;
      v33 = (__int64)v34;
      if ( v34 )
      {
        *v34 = v34 + 2;
        v34[1] = 0x200000000LL;
        v34[8] = v34 + 10;
        v34[9] = 0x200000000LL;
        if ( v41 )
        {
          v37 = sub_22077B0(0x30u);
          v35 = v39;
          if ( v37 )
          {
            *(_DWORD *)(v37 + 8) = 0;
            *(_QWORD *)(v37 + 16) = 0;
            *(_QWORD *)(v37 + 24) = v37 + 8;
            *(_QWORD *)(v37 + 32) = v37 + 8;
            *(_QWORD *)(v37 + 40) = 0;
          }
          *(_QWORD *)(v33 + 96) = v37;
        }
        else
        {
          v34[12] = 0;
        }
      }
      *(_QWORD *)(v35[53] + 8LL * v18) = v33;
      sub_2E11710(v35, v33, v18);
    }
    v20 += 2;
    result = *(_QWORD *)(v21 + 48) + 112LL * *(unsigned int *)(v21 + 56);
    *(_QWORD *)(result - 16) = v33;
    v18 += *(__int16 *)(v20 - 2);
  }
  while ( *(_WORD *)(v20 - 2) );
  return result;
}
