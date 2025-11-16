// Function: sub_34AA090
// Address: 0x34aa090
//
void __fastcall sub_34AA090(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v7; // rdx
  unsigned __int64 *v8; // rcx
  unsigned __int64 v9; // r8
  __int64 v10; // r9
  unsigned int v11; // ebx
  unsigned int v12; // esi
  __int64 v13; // r9
  __int64 v14; // rax
  __int64 v15; // rcx
  unsigned __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned int v19; // esi
  unsigned __int64 *v20; // rax
  unsigned __int64 v21; // r13
  __int64 v22; // r8
  __int64 v23; // rdx
  unsigned __int64 *v24; // rax
  __int64 v25; // rax
  int v26; // ecx
  unsigned int v27; // esi
  __int64 v28; // rdx
  unsigned __int64 *v29; // rax
  unsigned __int64 v30; // r8
  unsigned __int64 v31; // rsi
  __int64 i; // rax
  __int64 v33; // rcx
  unsigned __int64 v34; // rdx
  __int64 v35; // r12
  unsigned __int64 v36; // [rsp+18h] [rbp-158h]
  unsigned __int64 v37; // [rsp+18h] [rbp-158h]
  unsigned __int64 v38; // [rsp+18h] [rbp-158h]
  __int64 v39; // [rsp+20h] [rbp-150h] BYREF
  _BYTE *v40; // [rsp+28h] [rbp-148h] BYREF
  __int64 v41; // [rsp+30h] [rbp-140h]
  _BYTE v42[72]; // [rsp+38h] [rbp-138h] BYREF
  __int64 v43; // [rsp+80h] [rbp-F0h] BYREF
  _BYTE *v44; // [rsp+88h] [rbp-E8h]
  __int64 v45; // [rsp+90h] [rbp-E0h]
  _BYTE v46[72]; // [rsp+98h] [rbp-D8h] BYREF
  __int64 v47; // [rsp+E0h] [rbp-90h] BYREF
  _BYTE *v48; // [rsp+E8h] [rbp-88h]
  __int64 v49; // [rsp+F0h] [rbp-80h]
  _BYTE v50[120]; // [rsp+F8h] [rbp-78h] BYREF

  v6 = a2 + 8;
  v40 = v42;
  v41 = 0x400000000LL;
  v39 = a2 + 8;
  sub_34A26E0((__int64)&v39, 0, a3, a4, a5, a6);
  v11 = *(_DWORD *)(v39 + 192);
  if ( v11 )
  {
    v7 = (unsigned int)v41;
    for ( i = (unsigned int)(v41 - 1); v11 > (unsigned int)i; LODWORD(v41) = v41 + 1 )
    {
      v33 = (__int64)v40;
      v34 = v7 + 1;
      v9 = *(_QWORD *)(*(_QWORD *)&v40[16 * i] + 8LL * *(unsigned int *)&v40[16 * i + 12]) & 0xFFFFFFFFFFFFFFC0LL;
      v35 = (*(_QWORD *)(*(_QWORD *)&v40[16 * i] + 8LL * *(unsigned int *)&v40[16 * i + 12]) & 0x3FLL) + 1;
      if ( v34 > HIDWORD(v41) )
      {
        v38 = *(_QWORD *)(*(_QWORD *)&v40[16 * i] + 8LL * *(unsigned int *)&v40[16 * i + 12]) & 0xFFFFFFFFFFFFFFC0LL;
        sub_C8D5F0((__int64)&v40, v42, v34, 0x10u, v9, v10);
        v33 = (__int64)v40;
        v9 = v38;
      }
      v8 = (unsigned __int64 *)(16LL * (unsigned int)v41 + v33);
      *v8 = v9;
      v8[1] = v35;
      i = (unsigned int)v41;
      v7 = (unsigned int)(v41 + 1);
    }
  }
  v12 = *(_DWORD *)(a2 + 204);
  v44 = v46;
  v43 = v6;
  v45 = 0x400000000LL;
  sub_34A26E0((__int64)&v43, v12, v7, (__int64)v8, v9, v10);
  v14 = (unsigned int)v41;
  while ( (_DWORD)v14 && *((_DWORD *)v40 + 3) < *((_DWORD *)v40 + 2) )
  {
    v15 = (__int64)&v40[16 * v14 - 16];
    v16 = (unsigned __int64)&v44[16 * (unsigned int)v45 - 16];
    v17 = *(unsigned int *)(v15 + 12);
    v18 = *(_QWORD *)v15;
    if ( (_DWORD)v17 == *(_DWORD *)(v16 + 12) && *(_QWORD *)v16 == v18 )
      goto LABEL_18;
LABEL_7:
    v19 = *(_DWORD *)(a1 + 200);
    v20 = (unsigned __int64 *)(v18 + 16 * v17);
    v21 = *v20;
    v22 = v20[1];
    if ( v19 )
    {
      v31 = *v20;
      v49 = 0x400000000LL;
      v47 = a1 + 8;
      v37 = v22;
      v48 = v50;
      sub_34A3C90((__int64)&v47, v31, v18, v15, v22, v13);
      v30 = v37;
    }
    else
    {
      v23 = *(unsigned int *)(a1 + 204);
      if ( (_DWORD)v23 != 11 )
      {
        if ( (_DWORD)v23 )
        {
          v24 = (unsigned __int64 *)(a1 + 16);
          do
          {
            if ( *v24 >= v21 )
              break;
            ++v19;
            v24 += 2;
          }
          while ( (_DWORD)v23 != v19 );
        }
        LODWORD(v47) = v19;
        *(_DWORD *)(a1 + 204) = sub_34A32D0(a1 + 8, (unsigned int *)&v47, v23, v21, v22, 0);
        goto LABEL_14;
      }
      v47 = a1 + 8;
      v29 = (unsigned __int64 *)(a1 + 16);
      v48 = v50;
      v49 = 0x400000000LL;
      do
      {
        if ( *v29 >= v21 )
          break;
        ++v19;
        v29 += 2;
      }
      while ( v19 != 11 );
      v36 = v22;
      sub_34A26E0((__int64)&v47, v19, v23, v15, v22, v13);
      v30 = v36;
    }
    sub_34A8E00((__int64)&v47, v21, v30, 0);
    if ( v48 != v50 )
      _libc_free((unsigned __int64)v48);
LABEL_14:
    v25 = (__int64)&v40[16 * (unsigned int)v41 - 16];
    v26 = *(_DWORD *)(v25 + 12) + 1;
    *(_DWORD *)(v25 + 12) = v26;
    v14 = (unsigned int)v41;
    if ( v26 == *(_DWORD *)&v40[16 * (unsigned int)v41 - 8] )
    {
      v27 = *(_DWORD *)(v39 + 192);
      if ( v27 )
      {
        sub_F03D40((__int64 *)&v40, v27);
        v14 = (unsigned int)v41;
      }
    }
  }
  if ( (_DWORD)v45 )
  {
    v15 = *((unsigned int *)v44 + 2);
    if ( *((_DWORD *)v44 + 3) < (unsigned int)v15 )
    {
      v28 = (__int64)&v40[16 * v14 - 16];
      v17 = *(unsigned int *)(v28 + 12);
      v18 = *(_QWORD *)v28;
      goto LABEL_7;
    }
  }
LABEL_18:
  if ( v44 != v46 )
    _libc_free((unsigned __int64)v44);
  if ( v40 != v42 )
    _libc_free((unsigned __int64)v40);
}
