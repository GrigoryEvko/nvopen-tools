// Function: sub_10A7AD0
// Address: 0x10a7ad0
//
__int64 __fastcall sub_10A7AD0(__int64 *a1, int a2, unsigned __int8 *a3)
{
  _BYTE *v5; // rax
  _BYTE *v6; // rax
  char v7; // cl
  _BYTE *v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rax
  unsigned __int64 v11; // rax
  __int64 v12; // rcx
  _BYTE *v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  _BYTE *v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rax
  unsigned __int64 v22; // rax
  __int64 v23; // rcx
  _BYTE *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rcx
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rcx
  unsigned __int64 v38; // rax
  __int64 v39; // rdx
  unsigned __int8 *v40; // [rsp-60h] [rbp-60h]
  unsigned __int8 *v41; // [rsp-60h] [rbp-60h]
  unsigned __int8 *v42; // [rsp-60h] [rbp-60h]
  unsigned __int8 *v43; // [rsp-60h] [rbp-60h]

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = (_BYTE *)*((_QWORD *)a3 - 8);
  if ( *v5 != 69 )
    goto LABEL_4;
  v8 = (_BYTE *)*((_QWORD *)v5 - 4);
  if ( *v8 != 82 )
    goto LABEL_4;
  v9 = *((_QWORD *)v8 - 8);
  if ( v9 )
  {
    *(_QWORD *)a1[1] = v9;
    v10 = *((_QWORD *)v8 - 4);
    if ( v10 )
    {
      *(_QWORD *)a1[2] = v10;
      if ( *a1 )
      {
        v40 = a3;
        v11 = sub_B53900((__int64)v8);
        v12 = *a1;
        *(_DWORD *)v12 = v11;
        *(_BYTE *)(v12 + 4) = BYTE4(v11);
        a3 = v40;
      }
      goto LABEL_12;
    }
LABEL_4:
    v6 = (_BYTE *)*((_QWORD *)a3 - 4);
    v7 = *v6;
    goto LABEL_5;
  }
  v30 = *((_QWORD *)v8 - 4);
  if ( !v30 )
    goto LABEL_4;
  *(_QWORD *)a1[1] = v30;
  v31 = *((_QWORD *)v8 - 8);
  if ( !v31 )
    goto LABEL_4;
  *(_QWORD *)a1[2] = v31;
  if ( *a1 )
  {
    v42 = a3;
    v32 = sub_B53960((__int64)v8);
    v33 = *a1;
    *(_DWORD *)v33 = v32;
    *(_BYTE *)(v33 + 4) = BYTE4(v32);
    a3 = v42;
  }
LABEL_12:
  v6 = (_BYTE *)*((_QWORD *)a3 - 4);
  v7 = *v6;
  if ( *v6 != 68 )
  {
LABEL_5:
    if ( v7 != 69 )
      return 0;
    v19 = (_BYTE *)*((_QWORD *)v6 - 4);
    if ( *v19 != 82 )
      return 0;
    v20 = *((_QWORD *)v19 - 8);
    if ( v20 )
    {
      *(_QWORD *)a1[1] = v20;
      v21 = *((_QWORD *)v19 - 4);
      if ( !v21 )
        return 0;
      *(_QWORD *)a1[2] = v21;
      if ( *a1 )
      {
        v41 = a3;
        v22 = sub_B53900((__int64)v19);
        v23 = *a1;
        *(_DWORD *)v23 = v22;
        *(_BYTE *)(v23 + 4) = BYTE4(v22);
        a3 = v41;
      }
    }
    else
    {
      v34 = *((_QWORD *)v19 - 4);
      if ( !v34 )
        return 0;
      *(_QWORD *)a1[1] = v34;
      v35 = *((_QWORD *)v19 - 8);
      if ( !v35 )
        return 0;
      *(_QWORD *)a1[2] = v35;
      if ( *a1 )
      {
        v43 = a3;
        v36 = sub_B53960((__int64)v19);
        v37 = *a1;
        *(_DWORD *)v37 = v36;
        *(_BYTE *)(v37 + 4) = BYTE4(v36);
        a3 = v43;
      }
    }
    v24 = (_BYTE *)*((_QWORD *)a3 - 8);
    if ( *v24 == 68 )
    {
      v13 = (_BYTE *)*((_QWORD *)v24 - 4);
      if ( *v13 == 82 )
      {
        v25 = *((_QWORD *)v13 - 8);
        v26 = *((_QWORD *)v13 - 4);
        v27 = *(_QWORD *)a1[4];
        if ( v25 == v27 && v26 == *(_QWORD *)a1[5] )
        {
          if ( a1[3] )
            goto LABEL_42;
          return 1;
        }
        if ( v26 == v27 && v25 == *(_QWORD *)a1[5] )
        {
          if ( a1[3] )
          {
            v28 = sub_B53960((__int64)v13);
            v29 = a1[3];
            *(_DWORD *)v29 = v28;
            *(_BYTE *)(v29 + 4) = BYTE4(v28);
          }
          return 1;
        }
      }
    }
    return 0;
  }
  v13 = (_BYTE *)*((_QWORD *)v6 - 4);
  if ( *v13 != 82 )
    return 0;
  v14 = *((_QWORD *)v13 - 8);
  v15 = *((_QWORD *)v13 - 4);
  v16 = *(_QWORD *)a1[4];
  if ( v14 != v16 || v15 != *(_QWORD *)a1[5] )
  {
    if ( v15 == v16 && v14 == *(_QWORD *)a1[5] )
    {
      if ( a1[3] )
      {
        v17 = sub_B53960((__int64)v13);
        v18 = a1[3];
        *(_DWORD *)v18 = v17;
        *(_BYTE *)(v18 + 4) = BYTE4(v17);
        return 1;
      }
      return 1;
    }
    return 0;
  }
  if ( !a1[3] )
    return 1;
LABEL_42:
  v38 = sub_B53900((__int64)v13);
  v39 = a1[3];
  *(_DWORD *)v39 = v38;
  *(_BYTE *)(v39 + 4) = BYTE4(v38);
  return 1;
}
