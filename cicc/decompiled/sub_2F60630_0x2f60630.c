// Function: sub_2F60630
// Address: 0x2f60630
//
void __fastcall sub_2F60630(__int64 a1, unsigned __int16 ***a2)
{
  __int64 v3; // r9
  __int64 (*v4)(void); // rax
  __int64 v5; // rax
  unsigned __int16 *v6; // r13
  __int64 v7; // rdx
  unsigned __int8 v8; // r15
  __int64 v9; // r8
  __int64 v10; // rbx
  int v11; // edi
  __int16 *v12; // rax
  unsigned int v13; // edx
  __int64 v14; // rdx
  bool v15; // zf
  unsigned int v16; // r10d
  unsigned __int16 v17; // r14
  __int64 v18; // rsi
  unsigned __int8 v19; // cl
  __int64 v20; // rdx
  __int64 v21; // rsi
  __int64 (*v22)(); // rax
  __int64 v23; // rax
  char v24; // r14
  char v25; // r15
  unsigned __int16 *v26; // rax
  unsigned __int16 *v27; // rsi
  char v28; // cl
  unsigned __int16 v29; // dx
  __int64 v30; // rdi
  __int64 (__fastcall *v31)(__int64, __int64); // rax
  char v32; // al
  __int64 v33; // rax
  __int64 v34; // rcx
  _DWORD *v35; // rbx
  __int64 v36; // rax
  unsigned __int64 v37; // rdi
  __int64 v38; // [rsp+0h] [rbp-B0h]
  unsigned int v39; // [rsp+8h] [rbp-A8h]
  __int64 v40; // [rsp+8h] [rbp-A8h]
  unsigned __int8 v41; // [rsp+14h] [rbp-9Ch]
  unsigned int v42; // [rsp+14h] [rbp-9Ch]
  __int64 v44; // [rsp+28h] [rbp-88h]
  unsigned __int16 *v45; // [rsp+30h] [rbp-80h]
  __int64 v46; // [rsp+30h] [rbp-80h]
  __int64 v47; // [rsp+38h] [rbp-78h]
  char v48; // [rsp+38h] [rbp-78h]
  __int16 v49; // [rsp+38h] [rbp-78h]
  __int64 v50; // [rsp+38h] [rbp-78h]
  unsigned __int16 *v51; // [rsp+40h] [rbp-70h] BYREF
  __int64 v52; // [rsp+48h] [rbp-68h]
  unsigned __int64 v53; // [rsp+50h] [rbp-60h]
  _BYTE v54[88]; // [rsp+58h] [rbp-58h] BYREF

  v3 = *(_QWORD *)a1 + 24LL * *((unsigned __int16 *)*a2 + 12);
  v44 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL);
  if ( !*(_QWORD *)(v3 + 16) )
  {
    v50 = *(_QWORD *)a1 + 24LL * *((unsigned __int16 *)*a2 + 12);
    v36 = sub_2207820(2LL * *((unsigned __int16 *)*a2 + 10));
    v3 = v50;
    v37 = *(_QWORD *)(v50 + 16);
    *(_QWORD *)(v50 + 16) = v36;
    if ( v37 )
    {
      j_j___libc_free_0_0(v37);
      v3 = v50;
    }
  }
  v52 = 0;
  v51 = (unsigned __int16 *)v54;
  v53 = 16;
  v4 = (__int64 (*)(void))a2[8];
  if ( v4 )
  {
    v47 = v3;
    v5 = v4();
    v3 = v47;
    v6 = (unsigned __int16 *)v5;
    v45 = (unsigned __int16 *)(v5 + 2 * v7);
    if ( v45 != (unsigned __int16 *)v5 )
    {
LABEL_4:
      v8 = -1;
      v9 = 0;
      v10 = 0;
      v48 = -1;
      while ( 1 )
      {
        while ( 1 )
        {
          v16 = *v6;
          v17 = *v6;
          if ( (*(_QWORD *)(*(_QWORD *)(a1 + 224) + 8LL * (v16 >> 6)) & (1LL << *v6)) == 0 )
            break;
LABEL_9:
          if ( v45 == ++v6 )
            goto LABEL_19;
        }
        v18 = *(_QWORD *)(a1 + 24);
        v19 = *(_BYTE *)(*(_QWORD *)(a1 + 304) + (unsigned __int16)v16);
        v20 = *(_QWORD *)(v18 + 8);
        v21 = *(_QWORD *)(v18 + 56);
        if ( v8 > v19 )
          v8 = *(_BYTE *)(*(_QWORD *)(a1 + 304) + (unsigned __int16)v16);
        v12 = (__int16 *)(v21 + 2LL * (*(_DWORD *)(v20 + 24LL * (unsigned __int16)v16 + 16) >> 12));
        v13 = *(_DWORD *)(v20 + 24LL * (unsigned __int16)v16 + 16) & 0xFFF;
        if ( !v12 )
          goto LABEL_6;
        while ( !*(_WORD *)(*(_QWORD *)(a1 + 88) + 2LL * v13) )
        {
          v11 = *v12++;
          v13 += v11;
          if ( !(_WORD)v11 )
            goto LABEL_6;
        }
        v22 = *(__int64 (**)())(*(_QWORD *)v44 + 456LL);
        if ( v22 != sub_2F5FDF0 )
        {
          v38 = v3;
          v39 = v9;
          v41 = *(_BYTE *)(*(_QWORD *)(a1 + 304) + (unsigned __int16)v16);
          v32 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD))v22)(v44, *(_QWORD *)(a1 + 16), *v6);
          v19 = v41;
          v9 = v39;
          v3 = v38;
          if ( v32 )
          {
LABEL_6:
            v14 = (unsigned int)v10;
            v15 = v19 == (unsigned __int8)v48;
            v48 = v19;
            if ( !v15 )
              v9 = (unsigned int)v10;
            v10 = (unsigned int)(v10 + 1);
            *(_WORD *)(*(_QWORD *)(v3 + 16) + 2 * v14) = v17;
            goto LABEL_9;
          }
        }
        v23 = v52;
        if ( v52 + 1 > v53 )
        {
          v40 = v3;
          v42 = v9;
          sub_C8D290((__int64)&v51, v54, v52 + 1, 2u, v9, v3);
          v23 = v52;
          v3 = v40;
          v9 = v42;
        }
        ++v6;
        v51[v23] = v17;
        ++v52;
        if ( v45 == v6 )
        {
LABEL_19:
          v24 = v8;
          v25 = v48;
          goto LABEL_20;
        }
      }
    }
  }
  else
  {
    v6 = **a2;
    v45 = &v6[*((unsigned __int16 *)*a2 + 10)];
    if ( v45 != v6 )
      goto LABEL_4;
  }
  v24 = -1;
  LOWORD(v9) = 0;
  v25 = -1;
  v10 = 0;
LABEL_20:
  *(_DWORD *)(v3 + 4) = v10 + v52;
  v26 = v51;
  v27 = &v51[v52];
  if ( v27 != v51 )
  {
    while ( 1 )
    {
      v28 = v25;
      v25 = *(_BYTE *)(*(_QWORD *)(a1 + 304) + *v26);
      v29 = *v26;
      if ( v28 != v25 )
        LOWORD(v9) = v10;
      ++v26;
      *(_WORD *)(*(_QWORD *)(v3 + 16) + 2 * v10) = v29;
      if ( v27 == v26 )
        break;
      v10 = (unsigned int)(v10 + 1);
    }
  }
  if ( (_DWORD)qword_5024728 && *(_DWORD *)(v3 + 4) > (unsigned int)qword_5024728 )
    *(_DWORD *)(v3 + 4) = qword_5024728;
  v30 = *(_QWORD *)(a1 + 24);
  v31 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v30 + 352LL);
  if ( v31 != sub_2EBDF80 )
  {
    v46 = v3;
    v49 = v9;
    v33 = ((__int64 (__fastcall *)(__int64, unsigned __int16 ***, _QWORD))v31)(v30, a2, *(_QWORD *)(a1 + 16));
    LOWORD(v9) = v49;
    v3 = v46;
    if ( v33 )
    {
      if ( a2 != (unsigned __int16 ***)v33 )
      {
        v35 = (_DWORD *)(*(_QWORD *)a1 + 24LL * *(unsigned __int16 *)(*(_QWORD *)v33 + 24LL));
        if ( *(_DWORD *)(a1 + 8) == *v35 )
        {
          if ( *(_DWORD *)(v46 + 4) >= v35[1] )
            goto LABEL_30;
          goto LABEL_41;
        }
        sub_2F60630(a1, v33, 3LL * *(unsigned __int16 *)(*(_QWORD *)v33 + 24LL), v34);
        v3 = v46;
        LOWORD(v9) = v49;
        if ( *(_DWORD *)(v46 + 4) < v35[1] )
LABEL_41:
          *(_BYTE *)(v3 + 8) = 1;
      }
    }
  }
LABEL_30:
  *(_BYTE *)(v3 + 9) = v24;
  *(_WORD *)(v3 + 10) = v9;
  *(_DWORD *)v3 = *(_DWORD *)(a1 + 8);
  if ( v51 != (unsigned __int16 *)v54 )
    _libc_free((unsigned __int64)v51);
}
