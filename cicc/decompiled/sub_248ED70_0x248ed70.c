// Function: sub_248ED70
// Address: 0x248ed70
//
__int64 __fastcall sub_248ED70(__int64 a1, unsigned __int64 *a2, __int64 a3)
{
  __int64 result; // rax
  unsigned __int64 *v4; // r9
  __int64 v6; // rbx
  __int64 v7; // r12
  unsigned int v8; // edx
  __int64 v9; // rax
  unsigned int v10; // ecx
  bool v11; // cc
  unsigned int v12; // esi
  bool v13; // cc
  bool v14; // zf
  bool v15; // cc
  unsigned int v16; // edx
  int v17; // eax
  __int64 v18; // rdx
  unsigned __int64 v19; // rax
  unsigned int v20; // esi
  unsigned int v21; // edx
  _DWORD *v22; // rcx
  unsigned __int64 *v23; // r13
  unsigned __int64 v24; // rax
  int v25; // edx
  __int64 v26; // rsi
  __int64 v27; // rdx
  unsigned __int64 *v28; // r15
  unsigned int v29; // edi
  unsigned int v30; // esi
  unsigned int v31; // esi
  bool v32; // cc
  bool v33; // cc
  unsigned __int64 v34; // rax
  __int64 v35; // rcx
  __int64 v36; // rdx
  unsigned int v37; // edi
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // rdx
  unsigned int v41; // edi
  unsigned __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // rax
  unsigned __int64 v45; // rcx
  unsigned __int64 v46; // r8
  unsigned int v47; // ecx
  unsigned int v48; // esi
  unsigned int v49; // eax
  _DWORD *v50; // [rsp-40h] [rbp-40h]

  result = (__int64)a2 - a1;
  if ( (__int64)a2 - a1 <= 256 )
    return result;
  v4 = a2;
  v6 = a3;
  if ( !a3 )
  {
    v28 = a2;
    goto LABEL_48;
  }
  v7 = a1 + 16;
  v50 = (_DWORD *)(a1 + 32);
  while ( 2 )
  {
    v8 = *(_DWORD *)(a1 + 16);
    --v6;
    v9 = a1 + 16 * (result >> 5);
    v10 = *(_DWORD *)v9;
    v11 = v8 <= *(_DWORD *)v9;
    if ( v8 >= *(_DWORD *)v9 )
    {
      if ( v8 != v10 || (v30 = *(_DWORD *)(v9 + 4), v11 = *(_DWORD *)(a1 + 20) <= v30, *(_DWORD *)(a1 + 20) >= v30) )
      {
        if ( !v11 || *(_QWORD *)(a1 + 24) >= *(_QWORD *)(v9 + 8) )
        {
          v12 = *((_DWORD *)v4 - 4);
          v13 = v8 <= v12;
          if ( v8 < v12
            || v8 == v12 && (v41 = *((_DWORD *)v4 - 3), v13 = *(_DWORD *)(a1 + 20) <= v41, *(_DWORD *)(a1 + 20) < v41) )
          {
            v42 = *(_QWORD *)(a1 + 24);
LABEL_46:
            v43 = *(_QWORD *)(a1 + 16);
            v20 = *(_DWORD *)a1;
            *(_QWORD *)(a1 + 16) = *(_QWORD *)a1;
            v44 = *(_QWORD *)(a1 + 8);
            *(_QWORD *)a1 = v43;
            *(_QWORD *)(a1 + 8) = v42;
            *(_QWORD *)(a1 + 24) = v44;
            goto LABEL_12;
          }
          if ( v13 )
          {
            v42 = *(_QWORD *)(a1 + 24);
            if ( v42 < *(v4 - 1) )
              goto LABEL_46;
            v14 = v10 == v12;
            v15 = v10 <= v12;
            if ( v10 < v12 )
              goto LABEL_11;
          }
          else
          {
            v14 = v10 == v12;
            v15 = v10 <= v12;
            if ( v10 < v12 )
              goto LABEL_11;
          }
          if ( v14 )
          {
            v48 = *((_DWORD *)v4 - 3);
            if ( *(_DWORD *)(v9 + 4) < v48 )
              goto LABEL_11;
            if ( *(_DWORD *)(v9 + 4) > v48 )
            {
LABEL_43:
              v38 = *(_QWORD *)a1;
              *(_QWORD *)a1 = *(_QWORD *)v9;
              v39 = *(_QWORD *)(v9 + 8);
              *(_QWORD *)v9 = v38;
              v40 = *(_QWORD *)(a1 + 8);
              *(_QWORD *)(a1 + 8) = v39;
              *(_QWORD *)(v9 + 8) = v40;
              v20 = *(_DWORD *)(a1 + 16);
              goto LABEL_12;
            }
          }
          else if ( !v15 )
          {
            goto LABEL_43;
          }
          if ( *(_QWORD *)(v9 + 8) < *(v4 - 1) )
            goto LABEL_11;
          goto LABEL_43;
        }
      }
    }
    v31 = *((_DWORD *)v4 - 4);
    v32 = v10 <= v31;
    if ( v10 < v31 )
      goto LABEL_43;
    if ( v10 == v31 )
    {
      v47 = *((_DWORD *)v4 - 3);
      v32 = *(_DWORD *)(v9 + 4) <= v47;
      if ( *(_DWORD *)(v9 + 4) < v47 )
        goto LABEL_43;
    }
    if ( v32 && *(_QWORD *)(v9 + 8) < *(v4 - 1) )
      goto LABEL_43;
    v33 = v8 <= v31;
    if ( v8 >= v31 )
    {
      if ( v8 != v31 || (v49 = *((_DWORD *)v4 - 3), v33 = *(_DWORD *)(a1 + 20) <= v49, *(_DWORD *)(a1 + 20) >= v49) )
      {
        v34 = *(_QWORD *)(a1 + 24);
        if ( !v33 || v34 >= *(v4 - 1) )
        {
          v35 = *(_QWORD *)(a1 + 16);
          v20 = *(_DWORD *)a1;
          *(_QWORD *)(a1 + 16) = *(_QWORD *)a1;
          v36 = *(_QWORD *)(a1 + 8);
          *(_QWORD *)a1 = v35;
          *(_QWORD *)(a1 + 8) = v34;
          *(_QWORD *)(a1 + 24) = v36;
          goto LABEL_12;
        }
      }
    }
LABEL_11:
    v16 = *(_DWORD *)a1;
    v17 = *(_DWORD *)(a1 + 4);
    *(_QWORD *)a1 = *(v4 - 2);
    *((_DWORD *)v4 - 4) = v16;
    v18 = *(v4 - 1);
    *((_DWORD *)v4 - 3) = v17;
    v19 = *(_QWORD *)(a1 + 8);
    *(_QWORD *)(a1 + 8) = v18;
    *(v4 - 1) = v19;
    v20 = *(_DWORD *)(a1 + 16);
LABEL_12:
    v21 = *(_DWORD *)a1;
    v22 = v50;
    v23 = (unsigned __int64 *)v7;
    v24 = (unsigned __int64)v4;
    while ( 1 )
    {
      v28 = v23;
      if ( v21 > v20 )
        goto LABEL_16;
      if ( v21 == v20 )
      {
        v37 = *(_DWORD *)(a1 + 4);
        if ( *(v22 - 3) < v37 )
          goto LABEL_16;
        if ( *(v22 - 3) > v37 )
          goto LABEL_20;
      }
      else if ( v21 < v20 )
      {
        goto LABEL_20;
      }
      if ( *((_QWORD *)v22 - 1) >= *(_QWORD *)(a1 + 8) )
        break;
LABEL_16:
      v20 = *v22;
      v23 += 2;
      v22 += 4;
    }
    while ( 1 )
    {
      do
LABEL_20:
        v24 -= 16LL;
      while ( *(_DWORD *)v24 > v21 );
      if ( *(_DWORD *)v24 != v21 )
      {
        if ( *(_DWORD *)v24 < v21 )
          goto LABEL_14;
        goto LABEL_24;
      }
      v29 = *(_DWORD *)(v24 + 4);
      if ( *(_DWORD *)(a1 + 4) >= v29 )
      {
        if ( *(_DWORD *)(a1 + 4) > v29 )
        {
LABEL_14:
          if ( (unsigned __int64)v23 >= v24 )
            goto LABEL_26;
LABEL_15:
          v25 = *(v22 - 3);
          *((_QWORD *)v22 - 2) = *(_QWORD *)v24;
          *(_DWORD *)v24 = v20;
          v26 = *(_QWORD *)(v24 + 8);
          *(_DWORD *)(v24 + 4) = v25;
          v27 = *((_QWORD *)v22 - 1);
          *((_QWORD *)v22 - 1) = v26;
          *(_QWORD *)(v24 + 8) = v27;
          v21 = *(_DWORD *)a1;
          goto LABEL_16;
        }
LABEL_24:
        if ( *(_QWORD *)(a1 + 8) >= *(_QWORD *)(v24 + 8) )
          break;
      }
    }
    if ( (unsigned __int64)v23 < v24 )
      goto LABEL_15;
LABEL_26:
    sub_248ED70(v23, v4, v6, v22);
    result = (__int64)v23 - a1;
    if ( (__int64)v23 - a1 > 256 )
    {
      if ( v6 )
      {
        v4 = v23;
        continue;
      }
LABEL_48:
      sub_EE1480(a1, (unsigned __int64)v28, (unsigned __int64)v28);
      do
      {
        v28 -= 2;
        v45 = *v28;
        v46 = v28[1];
        *v28 = *(_QWORD *)a1;
        v28[1] = *(_QWORD *)(a1 + 8);
        result = sub_2485ED0(a1, 0, ((__int64)v28 - a1) >> 4, v45, v46);
      }
      while ( (__int64)v28 - a1 > 16 );
    }
    return result;
  }
}
