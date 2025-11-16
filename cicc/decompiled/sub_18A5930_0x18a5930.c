// Function: sub_18A5930
// Address: 0x18a5930
//
_QWORD *__fastcall sub_18A5930(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  __int64 v4; // r14
  __int64 v5; // r15
  __int64 v6; // rcx
  __int64 v7; // r12
  _QWORD **v8; // rbx
  __int64 v9; // r8
  _QWORD *v10; // r13
  _QWORD **v11; // r9
  _QWORD *v12; // r10
  unsigned __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // r14
  unsigned int v16; // edx
  __int64 v17; // r11
  __int64 v18; // rax
  __int64 v19; // rbx
  __int64 v20; // r12
  unsigned __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rax
  unsigned int v24; // edx
  __int64 v25; // r15
  _QWORD *v26; // r14
  unsigned __int64 i; // r13
  __int64 v28; // rax
  __int64 v29; // rbx
  unsigned int v30; // ecx
  __int64 v31; // r12
  __int64 v32; // rbx
  unsigned __int64 j; // r12
  __int64 v34; // rax
  __int64 v35; // rbx
  unsigned int v36; // edx
  __int64 v37; // r9
  __int64 v38; // rbx
  __int64 v41; // [rsp+10h] [rbp-80h]
  __int64 v42; // [rsp+18h] [rbp-78h]
  _QWORD *v43; // [rsp+28h] [rbp-68h]
  __int64 v44; // [rsp+28h] [rbp-68h]
  _QWORD **v45; // [rsp+30h] [rbp-60h]
  _QWORD **v46; // [rsp+30h] [rbp-60h]
  _QWORD **v48; // [rsp+48h] [rbp-48h]
  __int64 v49; // [rsp+48h] [rbp-48h]
  __int64 v50; // [rsp+48h] [rbp-48h]
  __int64 v51; // [rsp+50h] [rbp-40h]
  _QWORD *v52; // [rsp+50h] [rbp-40h]
  __int64 v53; // [rsp+50h] [rbp-40h]
  __int64 v54; // [rsp+58h] [rbp-38h]
  __int64 v55; // [rsp+58h] [rbp-38h]

  v4 = a1;
  v41 = a3 & 1;
  v54 = (a3 - 1) / 2;
  if ( a2 < v54 )
  {
    v5 = a2;
    v6 = a1;
    while ( 1 )
    {
      v7 = 2 * (v5 + 1);
      v8 = (_QWORD **)(v6 + 16 * (v5 + 1));
      v9 = v7 - 1;
      v10 = *v8;
      v11 = (_QWORD **)(v6 + 8 * (v7 - 1));
      v12 = *v11;
      v13 = (*v8)[15];
      if ( (*v8)[9] )
      {
        v14 = v10[7];
        if ( !v13
          || (v15 = v10[13], v16 = *(_DWORD *)(v15 + 32), *(_DWORD *)(v14 + 32) < v16)
          || *(_DWORD *)(v14 + 32) == v16 && *(_DWORD *)(v14 + 36) < *(_DWORD *)(v15 + 36) )
        {
          v13 = *(_QWORD *)(v14 + 40);
          goto LABEL_11;
        }
      }
      else
      {
        if ( !v13 )
          goto LABEL_11;
        v15 = v10[13];
      }
      v17 = *(_QWORD *)(v15 + 64);
      v18 = v15 + 48;
      v13 = 0;
      if ( v17 != v18 )
      {
        v51 = v7 - 1;
        v48 = (_QWORD **)(v6 + 8 * (v7 - 1));
        v43 = *v11;
        v42 = v6;
        v45 = (_QWORD **)(v6 + 16 * (v5 + 1));
        v19 = v18;
        v20 = v17;
        do
        {
          v13 += sub_18A58D0(v20 + 64);
          v20 = sub_220EF30(v20);
        }
        while ( v19 != v20 );
        v9 = v51;
        v11 = v48;
        v8 = v45;
        v12 = v43;
        v7 = 2 * (v5 + 1);
        v6 = v42;
      }
LABEL_11:
      v21 = v12[15];
      if ( v12[9] )
      {
        v22 = v12[7];
        if ( v21 )
        {
          v23 = v12[13];
          v24 = *(_DWORD *)(v23 + 32);
          if ( *(_DWORD *)(v22 + 32) >= v24
            && (*(_DWORD *)(v22 + 32) != v24 || *(_DWORD *)(v22 + 36) >= *(_DWORD *)(v23 + 36)) )
          {
LABEL_15:
            v44 = v6;
            v46 = v11;
            v49 = v9;
            v52 = v12;
            v21 = sub_18A5060((__int64)v12);
            v6 = v44;
            v11 = v46;
            v9 = v49;
            v12 = v52;
            goto LABEL_16;
          }
        }
        v21 = *(_QWORD *)(v22 + 40);
      }
      else if ( v21 )
      {
        goto LABEL_15;
      }
LABEL_16:
      if ( v21 < v13 )
      {
        v10 = v12;
        v8 = v11;
        v7 = v9;
      }
      *(_QWORD *)(v6 + 8 * v5) = v10;
      if ( v7 >= v54 )
      {
        v4 = v6;
        if ( v41 )
          goto LABEL_29;
        goto LABEL_48;
      }
      v5 = v7;
    }
  }
  v8 = (_QWORD **)(a1 + 8 * a2);
  if ( (a3 & 1) == 0 )
  {
    v7 = a2;
LABEL_48:
    if ( (a3 - 2) / 2 == v7 )
    {
      v7 = 2 * v7 + 1;
      *v8 = *(_QWORD **)(v4 + 8 * v7);
      v8 = (_QWORD **)(v4 + 8 * v7);
    }
LABEL_29:
    if ( v7 > a2 )
    {
      v53 = v7;
      v25 = (v7 - 1) / 2;
      v55 = v4;
      while ( 1 )
      {
        v26 = *(_QWORD **)(v55 + 8 * v25);
        i = v26[15];
        if ( v26[9] )
        {
          v28 = v26[7];
          if ( !i
            || (v29 = v26[13], v30 = *(_DWORD *)(v29 + 32), *(_DWORD *)(v28 + 32) < v30)
            || *(_DWORD *)(v28 + 32) == v30 && *(_DWORD *)(v28 + 36) < *(_DWORD *)(v29 + 36) )
          {
            i = *(_QWORD *)(v28 + 40);
            goto LABEL_37;
          }
        }
        else
        {
          if ( !i )
            goto LABEL_37;
          v29 = v26[13];
        }
        v31 = *(_QWORD *)(v29 + 64);
        v32 = v29 + 48;
        for ( i = 0; v32 != v31; v31 = sub_220EF30(v31) )
          i += sub_18A58D0(v31 + 64);
LABEL_37:
        j = a4[15];
        if ( a4[9] )
        {
          v34 = a4[7];
          if ( !j
            || (v35 = a4[13], v36 = *(_DWORD *)(v35 + 32), *(_DWORD *)(v34 + 32) < v36)
            || *(_DWORD *)(v34 + 32) == v36 && *(_DWORD *)(v34 + 36) < *(_DWORD *)(v35 + 36) )
          {
            j = *(_QWORD *)(v34 + 40);
            goto LABEL_43;
          }
        }
        else
        {
          if ( !j )
            goto LABEL_43;
          v35 = a4[13];
        }
        v37 = *(_QWORD *)(v35 + 64);
        v38 = v35 + 48;
        for ( j = 0; v38 != v37; v37 = sub_220EF30(v50) )
        {
          v50 = v37;
          j += sub_18A58D0(v37 + 64);
        }
LABEL_43:
        v8 = (_QWORD **)(v55 + 8 * v53);
        if ( j >= i )
          break;
        *v8 = v26;
        v53 = v25;
        if ( a2 >= v25 )
        {
          v8 = (_QWORD **)(v55 + 8 * v25);
          break;
        }
        v25 = (v25 - 1) / 2;
      }
    }
  }
  *v8 = a4;
  return a4;
}
