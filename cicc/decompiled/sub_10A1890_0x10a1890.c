// Function: sub_10A1890
// Address: 0x10a1890
//
__int64 __fastcall sub_10A1890(__int64 a1, __int64 a2, unsigned __int8 *a3, __int64 a4, bool a5)
{
  __int64 v5; // r15
  __int64 v6; // r13
  unsigned __int8 v8; // al
  unsigned __int8 v9; // dl
  unsigned __int8 *v10; // r14
  unsigned __int8 v11; // al
  __int64 v12; // r14
  unsigned __int8 *v14; // r14
  _BYTE *v15; // r15
  _BYTE *v16; // rax
  unsigned int **v17; // r14
  _BOOL8 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rsi
  __int64 v21; // r13
  __int64 *v22; // r15
  __int64 v23; // r14
  unsigned int v24; // ebx
  unsigned int v25; // ebx
  __int64 v26; // rdx
  int v27; // r12d
  __int64 v28; // rbx
  __int64 v29; // r12
  __int64 v30; // rdx
  unsigned int v31; // esi
  unsigned __int8 v32; // r13
  _BYTE *v33; // r14
  char v34; // r13
  __int64 *v35; // r15
  __int64 v36; // r14
  __int64 v37; // r15
  __int64 v38; // rdx
  unsigned int v39; // esi
  unsigned int *v40; // r15
  __int64 v41; // r14
  __int64 v42; // rdx
  char v43; // [rsp+3h] [rbp-9Dh]
  char v44; // [rsp+4h] [rbp-9Ch]
  char v45; // [rsp+5h] [rbp-9Bh]
  char v46; // [rsp+6h] [rbp-9Ah]
  bool v47; // [rsp+7h] [rbp-99h]
  __int64 v49; // [rsp+8h] [rbp-98h]
  __int64 v50; // [rsp+8h] [rbp-98h]
  _QWORD v51[4]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v52; // [rsp+30h] [rbp-70h]
  _BYTE v53[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v54; // [rsp+60h] [rbp-40h]

  v5 = a2;
  v6 = (__int64)a3;
  v8 = *(_BYTE *)a2;
  v47 = a5;
  if ( *(_BYTE *)a2 > 0x1Cu )
  {
    if ( v8 != 63 )
      goto LABEL_3;
    goto LABEL_7;
  }
  if ( v8 != 5 )
  {
LABEL_3:
    v9 = *a3;
    if ( v9 > 0x1Cu )
      goto LABEL_4;
    goto LABEL_16;
  }
  if ( *(_WORD *)(a2 + 2) == 34 )
  {
LABEL_7:
    v45 = 0;
    goto LABEL_8;
  }
  v9 = *a3;
  if ( v9 > 0x1Cu )
  {
LABEL_4:
    if ( v9 != 63 )
      goto LABEL_5;
    goto LABEL_18;
  }
LABEL_16:
  if ( v9 != 5 || *(_WORD *)(v6 + 2) != 34 )
  {
LABEL_5:
    if ( v8 <= 0x1Cu )
    {
      if ( v8 != 5 || *(_WORD *)(a2 + 2) != 34 )
        return 0;
    }
    else if ( v8 != 63 )
    {
      return 0;
    }
    goto LABEL_7;
  }
LABEL_18:
  v45 = 1;
  v5 = v6;
  v6 = a2;
LABEL_8:
  v10 = sub_BD3990(*(unsigned __int8 **)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF)), a2);
  if ( v10 != sub_BD3990((unsigned __int8 *)v6, a2) )
  {
    v11 = *(_BYTE *)v6;
    if ( *(_BYTE *)v6 <= 0x1Cu )
    {
      if ( v11 != 5 || *(_WORD *)(v6 + 2) != 34 )
        return 0;
    }
    else if ( v11 != 63 )
    {
      return 0;
    }
    v14 = sub_BD3990(*(unsigned __int8 **)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF)), a2);
    if ( v14 != sub_BD3990(*(unsigned __int8 **)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF)), a2) )
      return 0;
    v46 = *(_BYTE *)(v5 + 1) >> 1;
    v15 = sub_F20BF0(a1, v5, 1);
    v43 = *(_BYTE *)(v6 + 1) >> 1;
    v16 = sub_F20BF0(a1, v6, 1);
    v17 = *(unsigned int ***)(a1 + 32);
    v44 = v46 & 1;
    if ( (v46 & 1) != 0 )
    {
      v18 = 0;
      v19 = v43 & 1;
      v44 = v43 & 1;
      if ( !a5 )
        goto LABEL_24;
    }
    else
    {
      v19 = 0;
      v18 = 0;
      if ( !a5 )
      {
LABEL_24:
        v20 = 15;
        v52 = 259;
        v51[0] = "gepdiff";
        v49 = (__int64)v16;
        v21 = (*(__int64 (__fastcall **)(unsigned int *, __int64, _BYTE *, _BYTE *, _BOOL8, __int64))(*(_QWORD *)v17[10] + 32LL))(
                v17[10],
                15,
                v15,
                v16,
                v18,
                v19);
        if ( !v21 )
        {
          v54 = 257;
          v21 = sub_B504D0(15, (__int64)v15, v49, (__int64)v53, 0, 0);
          v20 = v21;
          (*(void (__fastcall **)(unsigned int *, __int64, _QWORD *, unsigned int *, unsigned int *))(*(_QWORD *)v17[11] + 16LL))(
            v17[11],
            v21,
            v51,
            v17[7],
            v17[8]);
          v40 = *v17;
          v41 = (__int64)&(*v17)[4 * *((unsigned int *)v17 + 2)];
          while ( (unsigned int *)v41 != v40 )
          {
            v42 = *((_QWORD *)v40 + 1);
            v20 = *v40;
            v40 += 4;
            sub_B99FD0(v21, v20, v42);
          }
          if ( v47 )
          {
            v20 = 1;
            sub_B447F0((unsigned __int8 *)v21, 1);
          }
          if ( v44 )
          {
            v20 = 1;
            sub_B44850((unsigned __int8 *)v21, 1);
          }
        }
        goto LABEL_25;
      }
    }
    v47 = 0;
    if ( (v46 & 4) != 0 )
    {
      v18 = (v43 & 4) != 0;
      v47 = (v43 & 4) != 0;
    }
    goto LABEL_24;
  }
  v20 = v5;
  v32 = *(_BYTE *)(v5 + 1);
  v33 = sub_F20BF0(a1, v5, 0);
  v34 = v32 >> 1;
  if ( *v33 > 0x1Cu && a5 )
  {
    if ( !v45 )
    {
      if ( *v33 == 46 && (v34 & 1) != 0 )
        sub_B447F0(v33, 1);
      v21 = (__int64)v33;
      goto LABEL_26;
    }
    goto LABEL_49;
  }
  v21 = (__int64)v33;
LABEL_25:
  if ( v45 )
  {
    v33 = (_BYTE *)v21;
LABEL_49:
    v35 = *(__int64 **)(a1 + 32);
    v51[0] = "diff.neg";
    v52 = 259;
    v50 = sub_AD6530(*((_QWORD *)v33 + 1), v20);
    v21 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, _BYTE *, _QWORD, _QWORD))(*(_QWORD *)v35[10] + 32LL))(
            v35[10],
            15,
            v50,
            v33,
            0,
            0);
    if ( !v21 )
    {
      v54 = 257;
      v21 = sub_B504D0(15, v50, (__int64)v33, (__int64)v53, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)v35[11] + 16LL))(
        v35[11],
        v21,
        v51,
        v35[7],
        v35[8]);
      v36 = *v35;
      v37 = *v35 + 16LL * *((unsigned int *)v35 + 2);
      while ( v37 != v36 )
      {
        v38 = *(_QWORD *)(v36 + 8);
        v39 = *(_DWORD *)v36;
        v36 += 16;
        sub_B99FD0(v21, v39, v38);
      }
    }
  }
LABEL_26:
  v22 = *(__int64 **)(a1 + 32);
  v52 = 257;
  v23 = *(_QWORD *)(v21 + 8);
  v24 = sub_BCB060(v23);
  v25 = (unsigned int)sub_BCB060(a4) < v24 ? 38 : 40;
  if ( a4 == v23 )
    return v21;
  v12 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v22[10] + 120LL))(
          v22[10],
          v25,
          v21,
          a4);
  if ( !v12 )
  {
    v54 = 257;
    v12 = sub_B51D30(v25, v21, a4, (__int64)v53, 0, 0);
    if ( (unsigned __int8)sub_920620(v12) )
    {
      v26 = v22[12];
      v27 = *((_DWORD *)v22 + 26);
      if ( v26 )
        sub_B99FD0(v12, 3u, v26);
      sub_B45150(v12, v27);
    }
    (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)v22[11] + 16LL))(
      v22[11],
      v12,
      v51,
      v22[7],
      v22[8]);
    v28 = *v22;
    v29 = *v22 + 16LL * *((unsigned int *)v22 + 2);
    if ( *v22 != v29 )
    {
      do
      {
        v30 = *(_QWORD *)(v28 + 8);
        v31 = *(_DWORD *)v28;
        v28 += 16;
        sub_B99FD0(v12, v31, v30);
      }
      while ( v29 != v28 );
    }
  }
  return v12;
}
