// Function: sub_15BA790
// Address: 0x15ba790
//
__int64 __fastcall sub_15BA790(__int64 *a1, int a2, __int64 a3, _QWORD *a4, __int64 a5, unsigned int a6, char a7)
{
  int v8; // r13d
  int v11; // eax
  __int64 v12; // r8
  __int64 result; // rax
  int v14; // r15d
  __int64 v15; // rax
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rcx
  int v20; // eax
  unsigned int v21; // r9d
  __int64 *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r10
  __int64 v25; // rsi
  _QWORD *v26; // rsi
  __int64 v27; // r8
  _QWORD *v28; // rdx
  _QWORD *v29; // rsi
  _QWORD *v30; // rdx
  __int64 v31; // rdx
  unsigned int v32; // [rsp+8h] [rbp-C8h]
  __int64 v33; // [rsp+8h] [rbp-C8h]
  int v34; // [rsp+1Ch] [rbp-B4h]
  _QWORD *v35; // [rsp+28h] [rbp-A8h]
  _QWORD *v36; // [rsp+30h] [rbp-A0h]
  __int64 v37; // [rsp+38h] [rbp-98h]
  __int64 *v38; // [rsp+38h] [rbp-98h]
  int v39; // [rsp+40h] [rbp-90h]
  __int64 v40; // [rsp+48h] [rbp-88h]
  _QWORD *v41; // [rsp+50h] [rbp-80h]
  __int64 v42; // [rsp+50h] [rbp-80h]
  __int64 v43; // [rsp+50h] [rbp-80h]
  __int16 v44; // [rsp+5Ch] [rbp-74h]
  int v45; // [rsp+6Ch] [rbp-64h] BYREF
  _QWORD *v46; // [rsp+70h] [rbp-60h] BYREF
  __int64 v47; // [rsp+78h] [rbp-58h]
  _QWORD *v48; // [rsp+80h] [rbp-50h]
  __int64 v49; // [rsp+88h] [rbp-48h]
  int v50; // [rsp+90h] [rbp-40h]
  int v51; // [rsp+94h] [rbp-3Ch] BYREF
  __int64 v52[7]; // [rsp+98h] [rbp-38h] BYREF

  v8 = (int)a1;
  v44 = a2;
  if ( a6 )
  {
    v14 = 0;
LABEL_6:
    v15 = *a1;
    v46 = (_QWORD *)a3;
    v42 = (__int64)a4;
    v16 = v15 + 624;
    v17 = sub_161E980(24, (unsigned int)(a5 + 1));
    v18 = v17;
    if ( v17 )
    {
      v19 = v42;
      v43 = v17;
      sub_1623D80(v17, v8, 8, a6, (unsigned int)&v46, 1, v19, a5);
      v18 = v43;
      *(_WORD *)(v43 + 2) = v44;
      *(_DWORD *)(v43 + 4) = v14;
    }
    return sub_15BA5D0(v18, a6, v16);
  }
  v46 = a4;
  v47 = a5;
  v41 = a4;
  v48 = 0;
  v49 = 0;
  v11 = sub_1607C40(a4, a5);
  v12 = *a1;
  v52[0] = a3;
  v50 = v11;
  a4 = v41;
  v51 = a2;
  v37 = v12;
  v39 = *(_DWORD *)(v12 + 648);
  v40 = *(_QWORD *)(v12 + 632);
  if ( !v39 )
    goto LABEL_3;
  v45 = v11;
  v20 = sub_15B64F0(&v45, &v51, v52);
  a4 = v41;
  v21 = (v39 - 1) & v20;
  v22 = (__int64 *)(v40 + 8LL * v21);
  v23 = *v22;
  if ( *v22 == -8 )
    goto LABEL_3;
  v34 = 1;
  v24 = v49;
  v35 = &v46[v47];
  v36 = &v48[v49];
  while ( 1 )
  {
    if ( v23 != -16 && v51 == *(unsigned __int16 *)(v23 + 2) )
    {
      v32 = *(_DWORD *)(v23 + 8);
      if ( v52[0] == *(_QWORD *)(v23 - 8LL * v32) && v50 == *(_DWORD *)(v23 + 4) )
      {
        v25 = v32 - 1;
        if ( !v47 )
        {
          if ( v24 != v25 )
            goto LABEL_14;
          v26 = (_QWORD *)(v23 + 8 * (1LL - v32));
          if ( v36 == v48 )
            goto LABEL_35;
          v33 = v24;
          v27 = v37;
          v28 = v48;
          v38 = v22;
          while ( *v28 == *v26 )
          {
            ++v28;
            ++v26;
            if ( v36 == v28 )
              goto LABEL_31;
          }
          goto LABEL_24;
        }
        if ( v47 == v25 )
          break;
      }
    }
LABEL_14:
    v21 = (v39 - 1) & (v34 + v21);
    v22 = (__int64 *)(v40 + 8LL * v21);
    v23 = *v22;
    if ( *v22 == -8 )
      goto LABEL_3;
    ++v34;
  }
  v29 = (_QWORD *)(v23 + 8 * (1LL - v32));
  if ( v35 != v46 )
  {
    v33 = v24;
    v27 = v37;
    v30 = v46;
    v38 = v22;
    while ( *v30 == *v29 )
    {
      ++v30;
      ++v29;
      if ( v35 == v30 )
      {
LABEL_31:
        v22 = v38;
        v31 = *(_QWORD *)(v27 + 632) + 8LL * *(unsigned int *)(v27 + 648);
        goto LABEL_32;
      }
    }
LABEL_24:
    v37 = v27;
    v24 = v33;
    goto LABEL_14;
  }
LABEL_35:
  v31 = *(_QWORD *)(v37 + 632) + 8LL * *(unsigned int *)(v37 + 648);
LABEL_32:
  if ( (__int64 *)v31 == v22 || (result = *v22) == 0 )
  {
LABEL_3:
    result = 0;
    if ( !a7 )
      return result;
    v14 = v50;
    goto LABEL_6;
  }
  return result;
}
