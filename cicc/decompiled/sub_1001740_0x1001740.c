// Function: sub_1001740
// Address: 0x1001740
//
__int64 __fastcall sub_1001740(int a1, __int64 a2, _BYTE *a3)
{
  __int64 v4; // rax
  int v5; // eax
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rdx
  _QWORD *v12; // rdi
  int v13; // ecx
  __int64 *v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rdx
  int v17; // ecx
  __int64 *v18; // rax
  __int64 v19; // rdx
  _QWORD *v20; // rdi
  int v21; // ecx
  __int64 *v22; // rax
  __int64 v23; // rdi
  __int64 v24; // rdx
  int v25; // ecx
  __int64 *v26; // rax
  __int64 v27; // [rsp-28h] [rbp-28h]
  __int64 v28; // [rsp-20h] [rbp-20h]
  __int64 v29; // [rsp-18h] [rbp-18h]
  __int64 v30; // [rsp-10h] [rbp-10h]

  if ( *(_BYTE *)a2 != 85 )
    return 0;
  v4 = *(_QWORD *)(a2 - 32);
  if ( !v4 || *(_BYTE *)v4 || *(_QWORD *)(v4 + 24) != *(_QWORD *)(a2 + 80) || (*(_BYTE *)(v4 + 33) & 0x20) == 0 )
    return 0;
  v5 = *(_DWORD *)(v4 + 36);
  if ( v5 == 359 )
  {
    if ( *a3 != 42 )
      return 0;
    v6 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    v7 = *(_QWORD *)(a2 + 32 * (1 - v6));
    v8 = *(_QWORD *)(a2 - 32 * v6);
    v9 = *((_QWORD *)a3 - 8);
    v10 = *((_QWORD *)a3 - 4);
    if ( (v8 != v9 || v7 != v10) && (v8 != v10 || v7 != v9) )
      return 0;
    if ( a1 != 35 )
    {
      if ( a1 == 36 )
      {
        v11 = *(_QWORD *)(a2 + 8);
        v12 = *(_QWORD **)v11;
        v13 = *(unsigned __int8 *)(v11 + 8);
        if ( (unsigned int)(v13 - 17) <= 1 )
        {
          BYTE4(v28) = (_BYTE)v13 == 18;
          LODWORD(v28) = *(_DWORD *)(v11 + 32);
          v14 = (__int64 *)sub_BCB2A0(v12);
          v15 = sub_BCE1B0(v14, v28);
          return sub_AD6450(v15);
        }
LABEL_29:
        v15 = sub_BCB2A0(v12);
        return sub_AD6450(v15);
      }
      return 0;
    }
    v24 = *(_QWORD *)(a2 + 8);
    v20 = *(_QWORD **)v24;
    v25 = *(unsigned __int8 *)(v24 + 8);
    if ( (unsigned int)(v25 - 17) <= 1 )
    {
      BYTE4(v27) = (_BYTE)v25 == 18;
      LODWORD(v27) = *(_DWORD *)(v24 + 32);
      v26 = (__int64 *)sub_BCB2A0(v20);
      v23 = sub_BCE1B0(v26, v27);
      return sub_AD6400(v23);
    }
    goto LABEL_32;
  }
  if ( v5 != 371
    || *a3 != 44
    || *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)) != *((_QWORD *)a3 - 8)
    || *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) != *((_QWORD *)a3 - 4) )
  {
    return 0;
  }
  if ( a1 != 37 )
  {
    if ( a1 == 34 )
    {
      v16 = *(_QWORD *)(a2 + 8);
      v12 = *(_QWORD **)v16;
      v17 = *(unsigned __int8 *)(v16 + 8);
      if ( (unsigned int)(v17 - 17) <= 1 )
      {
        BYTE4(v30) = (_BYTE)v17 == 18;
        LODWORD(v30) = *(_DWORD *)(v16 + 32);
        v18 = (__int64 *)sub_BCB2A0(v12);
        v15 = sub_BCE1B0(v18, v30);
        return sub_AD6450(v15);
      }
      goto LABEL_29;
    }
    return 0;
  }
  v19 = *(_QWORD *)(a2 + 8);
  v20 = *(_QWORD **)v19;
  v21 = *(unsigned __int8 *)(v19 + 8);
  if ( (unsigned int)(v21 - 17) > 1 )
  {
LABEL_32:
    v23 = sub_BCB2A0(v20);
    return sub_AD6400(v23);
  }
  BYTE4(v29) = (_BYTE)v21 == 18;
  LODWORD(v29) = *(_DWORD *)(v19 + 32);
  v22 = (__int64 *)sub_BCB2A0(v20);
  v23 = sub_BCE1B0(v22, v29);
  return sub_AD6400(v23);
}
