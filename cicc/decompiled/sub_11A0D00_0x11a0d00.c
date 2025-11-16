// Function: sub_11A0D00
// Address: 0x11a0d00
//
__int64 __fastcall sub_11A0D00(__int64 a1, unsigned int a2, __int64 *a3)
{
  char v5; // al
  __int64 v6; // rcx
  __int64 v7; // rbx
  unsigned __int8 *v8; // rdi
  __int64 v9; // rdx
  unsigned __int8 *v10; // r13
  _QWORD *v11; // rdx
  _BYTE *v12; // rax
  _BYTE *v14; // rax
  _BOOL4 v15; // r8d
  _BYTE *v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // r9
  unsigned int v19; // eax
  bool v20; // al
  unsigned int v21; // edx
  unsigned __int64 v22; // r13
  const void *v23; // r13
  unsigned int v24; // eax
  unsigned __int64 v25; // r10
  const void *v26; // r10
  bool v27; // r8
  __int64 v28; // rax
  __int64 v29; // r12
  __int64 v30; // rbx
  __int64 v31; // rdx
  __int64 v32; // rdx
  _BYTE *v33; // rax
  unsigned int v34; // eax
  bool v35; // al
  const void *v36; // rdi
  unsigned int v37; // [rsp+8h] [rbp-88h]
  const void *v38; // [rsp+8h] [rbp-88h]
  bool v39; // [rsp+8h] [rbp-88h]
  bool v40; // [rsp+10h] [rbp-80h]
  bool v41; // [rsp+10h] [rbp-80h]
  __int64 v42; // [rsp+10h] [rbp-80h]
  __int64 v43; // [rsp+10h] [rbp-80h]
  __int64 v44; // [rsp+18h] [rbp-78h]
  __int64 v45; // [rsp+18h] [rbp-78h]
  __int64 v46; // [rsp+18h] [rbp-78h]
  bool v47; // [rsp+18h] [rbp-78h]
  _BYTE *v48; // [rsp+18h] [rbp-78h]
  unsigned int v49; // [rsp+18h] [rbp-78h]
  const void *v50; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v51; // [rsp+28h] [rbp-68h]
  const void *v52; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v53; // [rsp+38h] [rbp-58h]
  const void *v54; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v55; // [rsp+48h] [rbp-48h]
  const void *v56; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v57; // [rsp+58h] [rbp-38h]

  v5 = *(_BYTE *)(a1 + 7) & 0x40;
  if ( v5 )
    v6 = *(_QWORD *)(a1 - 8);
  else
    v6 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  v7 = 32LL * a2;
  v8 = *(unsigned __int8 **)(v6 + v7);
  v9 = *v8;
  v10 = v8 + 24;
  if ( (_BYTE)v9 != 17 )
  {
    if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v8 + 1) + 8LL) - 17 > 1
      || (unsigned __int8)v9 > 0x15u
      || (v14 = sub_AD7630((__int64)v8, 0, v9)) == 0
      || *v14 != 17 )
    {
      return 0;
    }
    v10 = v14 + 24;
    v5 = *(_BYTE *)(a1 + 7) & 0x40;
  }
  if ( v5 )
    v11 = *(_QWORD **)(a1 - 8);
  else
    v11 = (_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  v12 = (_BYTE *)*v11;
  if ( *(_BYTE *)*v11 != 82 )
    return sub_11A0AE0(a1, a2, a3);
  v16 = (_BYTE *)*((_QWORD *)v12 - 8);
  if ( !v16 )
    return sub_11A0AE0(a1, a2, a3);
  v17 = *((_QWORD *)v12 - 4);
  v18 = v17 + 24;
  if ( *(_BYTE *)v17 != 17 )
  {
    v48 = (_BYTE *)*((_QWORD *)v12 - 8);
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v17 + 8) + 8LL) - 17 > 1 )
      return sub_11A0AE0(a1, a2, a3);
    if ( *(_BYTE *)v17 > 0x15u )
      return sub_11A0AE0(a1, a2, a3);
    v33 = sub_AD7630(v17, 0, (__int64)v16);
    if ( !v33 || *v33 != 17 )
      return sub_11A0AE0(a1, a2, a3);
    v16 = v48;
    v18 = (__int64)(v33 + 24);
  }
  if ( *v16 <= 0x15u )
    return sub_11A0AE0(a1, a2, a3);
  v19 = *(_DWORD *)(v18 + 8);
  if ( v19 != *((_DWORD *)v10 + 2) )
    return sub_11A0AE0(a1, a2, a3);
  if ( v19 <= 0x40 )
  {
    v15 = 0;
    if ( *(_QWORD *)v18 != *(_QWORD *)v10 )
    {
LABEL_23:
      v21 = *((_DWORD *)v10 + 2);
      v55 = v21;
      if ( v21 > 0x40 )
      {
        v43 = v18;
        sub_C43780((__int64)&v54, (const void **)v10);
        v21 = v55;
        v18 = v43;
        if ( v55 > 0x40 )
        {
          sub_C43B90(&v54, a3);
          v21 = v55;
          v23 = v54;
          v18 = v43;
LABEL_26:
          v55 = 0;
          v24 = *(_DWORD *)(v18 + 8);
          v57 = v21;
          v56 = v23;
          v51 = v24;
          if ( v24 > 0x40 )
          {
            v37 = v21;
            v42 = v18;
            sub_C43780((__int64)&v50, (const void **)v18);
            v18 = v42;
            v21 = v37;
            if ( v51 > 0x40 )
            {
              v49 = v37;
              sub_C43B90(&v50, a3);
              v34 = v51;
              v26 = v50;
              v51 = 0;
              v21 = v37;
              v18 = v42;
              v53 = v34;
              v52 = v50;
              if ( v34 > 0x40 )
              {
                v38 = v50;
                v35 = sub_C43C50((__int64)&v52, &v56);
                v21 = v49;
                v18 = v42;
                v27 = v35;
                if ( v38 )
                {
                  v36 = v38;
                  v39 = v35;
                  j_j___libc_free_0_0(v36);
                  v21 = v49;
                  v18 = v42;
                  v27 = v39;
                  if ( v51 > 0x40 )
                  {
                    if ( v50 )
                    {
                      j_j___libc_free_0_0(v50);
                      v27 = v39;
                      v18 = v42;
                      v21 = v49;
                    }
                  }
                }
LABEL_30:
                if ( v21 > 0x40 && v23 )
                {
                  v40 = v27;
                  v45 = v18;
                  j_j___libc_free_0_0(v23);
                  v27 = v40;
                  v18 = v45;
                }
                if ( v55 > 0x40 && v54 )
                {
                  v41 = v27;
                  v46 = v18;
                  j_j___libc_free_0_0(v54);
                  v27 = v41;
                  v18 = v46;
                }
                if ( v27 )
                {
                  v47 = v27;
                  v28 = sub_AD8D80(*(_QWORD *)(a1 + 8), v18);
                  v15 = v47;
                  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
                    v29 = *(_QWORD *)(a1 - 8);
                  else
                    v29 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
                  v30 = v29 + v7;
                  if ( *(_QWORD *)v30 )
                  {
                    v31 = *(_QWORD *)(v30 + 8);
                    **(_QWORD **)(v30 + 16) = v31;
                    if ( v31 )
                      *(_QWORD *)(v31 + 16) = *(_QWORD *)(v30 + 16);
                  }
                  *(_QWORD *)v30 = v28;
                  if ( v28 )
                  {
                    v32 = *(_QWORD *)(v28 + 16);
                    *(_QWORD *)(v30 + 8) = v32;
                    if ( v32 )
                      *(_QWORD *)(v32 + 16) = v30 + 8;
                    *(_QWORD *)(v30 + 16) = v28 + 16;
                    *(_QWORD *)(v28 + 16) = v30;
                  }
                  return v15;
                }
                return sub_11A0AE0(a1, a2, a3);
              }
LABEL_29:
              v27 = v23 == v26;
              goto LABEL_30;
            }
            v25 = (unsigned __int64)v50;
          }
          else
          {
            v25 = *(_QWORD *)v18;
          }
          v26 = (const void *)(*a3 & v25);
          goto LABEL_29;
        }
        v22 = (unsigned __int64)v54;
      }
      else
      {
        v22 = *(_QWORD *)v10;
      }
      v23 = (const void *)(*a3 & v22);
      v54 = v23;
      goto LABEL_26;
    }
  }
  else
  {
    v44 = v18;
    v20 = sub_C43C50(v18, (const void **)v10);
    v15 = 0;
    v18 = v44;
    if ( !v20 )
      goto LABEL_23;
  }
  return v15;
}
