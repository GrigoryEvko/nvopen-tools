// Function: sub_2C4EBF0
// Address: 0x2c4ebf0
//
__int64 __fastcall sub_2C4EBF0(
        unsigned int *a1,
        unsigned int *a2,
        unsigned int *a3,
        unsigned int *a4,
        __int64 a5,
        __int64 a6,
        __int64 **a7,
        _QWORD *a8)
{
  unsigned int *v8; // r9
  unsigned int *v9; // r10
  unsigned int *v11; // r12
  unsigned int *v12; // rbx
  unsigned int v14; // eax
  _BYTE *v15; // rdx
  __int64 v16; // r15
  __int64 v17; // r14
  unsigned __int8 *v18; // rax
  __int64 v19; // rsi
  char v20; // di
  __int64 v21; // rcx
  __int64 v22; // rsi
  unsigned int v23; // eax
  __int64 v24; // r9
  __int64 v25; // rcx
  __int64 v26; // rdx
  unsigned int v27; // eax
  __int64 result; // rax
  __int64 v29; // rcx
  __int64 v30; // rsi
  __int64 v31; // rdx
  unsigned int v32; // edi
  __int64 v33; // rcx
  _QWORD *v34; // r11
  _QWORD *v35; // rcx
  _QWORD *v36; // rax
  _QWORD *v37; // rcx
  __int64 v38; // rdi
  _QWORD *v39; // rax
  _QWORD *v40; // rdx
  __int64 *v41; // rax
  __int64 *v42; // rax
  _QWORD *v43; // [rsp+0h] [rbp-60h]
  _QWORD *v44; // [rsp+0h] [rbp-60h]
  unsigned int *v45; // [rsp+8h] [rbp-58h]
  unsigned int *v46; // [rsp+8h] [rbp-58h]
  unsigned int *v47; // [rsp+10h] [rbp-50h]
  unsigned int *v48; // [rsp+10h] [rbp-50h]
  _BYTE *v49; // [rsp+18h] [rbp-48h]
  _BYTE *v50; // [rsp+18h] [rbp-48h]
  __int64 *v51; // [rsp+28h] [rbp-38h]

  v8 = a2;
  v9 = a4;
  v11 = a3;
  v12 = a1;
  if ( a3 != a4 && a1 != a2 )
  {
    do
    {
      v15 = (_BYTE *)*a8;
      v16 = *v11;
      v17 = *v12;
      if ( *(_BYTE *)*a8 != 92 )
        goto LABEL_13;
      v51 = *a7;
      v18 = (unsigned __int8 *)*((_QWORD *)v15 - 4);
      if ( (unsigned int)*v18 - 12 > 1 || (v19 = *((_QWORD *)v15 - 8), v20 = *(_BYTE *)v19, *(_BYTE *)v19 != 92) )
      {
LABEL_9:
        v21 = *((_QWORD *)v15 + 9);
        LODWORD(v16) = *(_DWORD *)(v21 + 4 * v16);
        goto LABEL_10;
      }
      v33 = *v51;
      if ( *(_BYTE *)(*v51 + 28) )
      {
        v34 = *(_QWORD **)(v33 + 8);
        v35 = &v34[*(unsigned int *)(v33 + 20)];
        if ( v34 == v35 )
          goto LABEL_9;
        v36 = v35;
        v37 = v34;
        while ( v19 != *v37 )
        {
          if ( v36 == ++v37 )
            goto LABEL_41;
        }
      }
      else
      {
        v43 = a8;
        v45 = v9;
        v47 = v8;
        v49 = (_BYTE *)*a8;
        v41 = sub_C8CA60(*v51, v19);
        v15 = v49;
        v8 = v47;
        v9 = v45;
        a8 = v43;
        v20 = *v49;
        v51 = *a7;
        if ( !v41 )
        {
LABEL_41:
          v21 = *((_QWORD *)v15 + 9);
          LODWORD(v16) = *(_DWORD *)(v21 + 4 * v16);
          goto LABEL_32;
        }
      }
      v21 = *((_QWORD *)v15 + 9);
      LODWORD(v16) = *(_DWORD *)(*(_QWORD *)(v19 + 72) + 4LL * *(unsigned int *)(v21 + 4 * v16));
LABEL_32:
      if ( v20 != 92 )
        goto LABEL_13;
      v18 = (unsigned __int8 *)*((_QWORD *)v15 - 4);
LABEL_10:
      if ( (unsigned int)*v18 - 12 > 1 || (v22 = *((_QWORD *)v15 - 8), *(_BYTE *)v22 != 92) )
      {
LABEL_12:
        LODWORD(v17) = *(_DWORD *)(v21 + 4 * v17);
        goto LABEL_13;
      }
      v38 = *v51;
      if ( *(_BYTE *)(*v51 + 28) )
      {
        v39 = *(_QWORD **)(v38 + 8);
        v40 = &v39[*(unsigned int *)(v38 + 20)];
        if ( v39 == v40 )
          goto LABEL_12;
        while ( v22 != *v39 )
        {
          if ( v40 == ++v39 )
            goto LABEL_12;
        }
      }
      else
      {
        v44 = a8;
        v46 = v9;
        v48 = v8;
        v50 = v15;
        v42 = sub_C8CA60(v38, v22);
        v8 = v48;
        v9 = v46;
        a8 = v44;
        v21 = *((_QWORD *)v50 + 9);
        if ( !v42 )
          goto LABEL_12;
      }
      LODWORD(v17) = *(_DWORD *)(*(_QWORD *)(v22 + 72) + 4LL * *(unsigned int *)(v21 + 4 * v17));
LABEL_13:
      if ( (int)v16 < (int)v17 )
      {
        v14 = *v11;
        a5 += 8;
        v11 += 2;
        *(_DWORD *)(a5 - 8) = v14;
        *(_DWORD *)(a5 - 4) = *(v11 - 1);
        if ( v12 == v8 )
          break;
      }
      else
      {
        v23 = *v12;
        v12 += 2;
        a5 += 8;
        *(_DWORD *)(a5 - 8) = v23;
        *(_DWORD *)(a5 - 4) = *(v12 - 1);
        if ( v12 == v8 )
          break;
      }
    }
    while ( v11 != v9 );
  }
  v24 = (char *)v8 - (char *)v12;
  v25 = v24 >> 3;
  if ( v24 <= 0 )
  {
    result = a5;
  }
  else
  {
    v26 = a5;
    do
    {
      v27 = *v12;
      v26 += 8;
      v12 += 2;
      *(_DWORD *)(v26 - 8) = v27;
      *(_DWORD *)(v26 - 4) = *(v12 - 1);
      --v25;
    }
    while ( v25 );
    result = a5 + v24;
  }
  v29 = (char *)v9 - (char *)v11;
  v30 = ((char *)v9 - (char *)v11) >> 3;
  if ( (char *)v9 - (char *)v11 > 0 )
  {
    v31 = result;
    do
    {
      v32 = *v11;
      v31 += 8;
      v11 += 2;
      *(_DWORD *)(v31 - 8) = v32;
      *(_DWORD *)(v31 - 4) = *(v11 - 1);
      --v30;
    }
    while ( v30 );
    if ( v29 <= 0 )
      v29 = 8;
    result += v29;
  }
  return result;
}
