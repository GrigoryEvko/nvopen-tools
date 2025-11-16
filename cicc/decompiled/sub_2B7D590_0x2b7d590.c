// Function: sub_2B7D590
// Address: 0x2b7d590
//
__int64 ***__fastcall sub_2B7D590(_QWORD *a1, __int64 a2, char *a3, unsigned __int64 a4)
{
  __int64 v6; // r8
  __int64 *v7; // rdi
  __int64 v8; // r9
  __int64 v9; // rax
  int v10; // esi
  __int64 v11; // rax
  __int64 v12; // r9
  __int64 *v13; // rax
  __int64 v14; // rdx
  _OWORD *v15; // rcx
  unsigned __int64 v16; // rdx
  int v17; // r8d
  _OWORD *v18; // rax
  _OWORD *v19; // rdx
  __int64 v20; // rdx
  _QWORD *v21; // r8
  unsigned int *v22; // rax
  unsigned int *v23; // rdi
  __int64 v24; // rsi
  int v25; // edx
  __int64 ***v26; // r14
  char *v28; // [rsp+0h] [rbp-110h]
  unsigned __int64 v29; // [rsp+8h] [rbp-108h]
  __int64 v30; // [rsp+10h] [rbp-100h]
  int v31; // [rsp+10h] [rbp-100h]
  __int64 v32; // [rsp+18h] [rbp-F8h]
  unsigned __int64 v33; // [rsp+18h] [rbp-F8h]
  _OWORD *v34; // [rsp+20h] [rbp-F0h] BYREF
  __int64 v35; // [rsp+28h] [rbp-E8h]
  _OWORD v36[3]; // [rsp+30h] [rbp-E0h] BYREF
  __int64 *v37; // [rsp+60h] [rbp-B0h] BYREF
  char v38; // [rsp+68h] [rbp-A8h]
  _BYTE *v39; // [rsp+70h] [rbp-A0h]
  __int64 v40; // [rsp+78h] [rbp-98h]
  _BYTE v41[48]; // [rsp+80h] [rbp-90h] BYREF
  _BYTE *v42; // [rsp+B0h] [rbp-60h]
  __int64 v43; // [rsp+B8h] [rbp-58h]
  _BYTE v44[16]; // [rsp+C0h] [rbp-50h] BYREF
  __int64 v45; // [rsp+D0h] [rbp-40h]
  __int64 v46; // [rsp+D8h] [rbp-38h]

  v6 = a1[1];
  v7 = *(__int64 **)(*(_QWORD *)(a2 + 8) + 24LL);
  v8 = v6 + 3368;
  v9 = *(_QWORD *)(**(_QWORD **)*a1 + 8LL);
  if ( *(_BYTE *)(v9 + 8) == 17 )
  {
    v10 = *(_DWORD *)(v9 + 32);
    if ( v10 != 1 )
    {
      v28 = a3;
      v29 = a4;
      v30 = v6 + 3368;
      v32 = v6;
      v11 = sub_BCDA70(v7, v10);
      a3 = v28;
      a4 = v29;
      v8 = v30;
      v6 = v32;
      v7 = (__int64 *)v11;
    }
  }
  v46 = v6;
  v37 = v7;
  v40 = 0xC00000000LL;
  v43 = 0x200000000LL;
  v38 = 0;
  v39 = v41;
  v42 = v44;
  v45 = v8;
  sub_2B7BF50((__int64)&v37, a2, a3, a4);
  v13 = (__int64 *)a1[2];
  v14 = *v13;
  v35 = 0x300000000LL;
  v15 = v36;
  v34 = v36;
  v16 = *(unsigned int *)(v14 + 216);
  v17 = v16;
  if ( v16 )
  {
    v18 = v36;
    if ( v16 > 3 )
    {
      v31 = v16;
      v33 = v16;
      sub_C8D5F0((__int64)&v34, v36, v16, 0x10u, v16, v12);
      v15 = v34;
      v17 = v31;
      v18 = &v34[(unsigned int)v35];
      v19 = &v34[v33];
      if ( v19 != v18 )
        goto LABEL_7;
    }
    else
    {
      v19 = &v36[v16];
      if ( v19 != v36 )
      {
        do
        {
LABEL_7:
          if ( v18 )
          {
            *(_QWORD *)v18 = 0;
            *((_DWORD *)v18 + 2) = 0;
          }
          ++v18;
        }
        while ( v19 != v18 );
        v15 = v34;
      }
    }
    LODWORD(v35) = v17;
    v13 = (__int64 *)a1[2];
  }
  v20 = *v13;
  v21 = (_QWORD *)a1[1];
  v22 = *(unsigned int **)(*v13 + 208);
  v23 = &v22[2 * *(unsigned int *)(v20 + 216)];
  if ( v22 != v23 )
  {
    do
    {
      v24 = *v22;
      v22 += 2;
      ++v15;
      v25 = *(v22 - 1);
      *((_QWORD *)v15 - 2) = *(_QWORD *)(*v21 + 8 * v24);
      *((_DWORD *)v15 - 2) = v25;
    }
    while ( v23 != v22 );
    v15 = v34;
  }
  v26 = sub_2B7B8F0((__int64)&v37, 0, 0, (__int64)v15, (unsigned int)v35, 0, 0, 0, 0, 0);
  if ( v34 != v36 )
    _libc_free((unsigned __int64)v34);
  if ( v42 != v44 )
    _libc_free((unsigned __int64)v42);
  if ( v39 != v41 )
    _libc_free((unsigned __int64)v39);
  return v26;
}
