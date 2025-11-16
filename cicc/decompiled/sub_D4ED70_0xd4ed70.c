// Function: sub_D4ED70
// Address: 0xd4ed70
//
_QWORD **__fastcall sub_D4ED70(__int64 a1, __int64 a2, _QWORD **a3)
{
  _QWORD *v3; // rax
  __int64 v4; // r15
  __int64 v5; // r12
  _QWORD *v7; // rdx
  __int64 v8; // rcx
  _QWORD *i; // rax
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // rbx
  unsigned int v15; // r15d
  __int64 v16; // r14
  __int64 v17; // rdx
  __int64 v18; // rax
  int v19; // edi
  __int64 v20; // r10
  _QWORD *v21; // rax
  int v22; // edi
  unsigned int v23; // esi
  __int64 *v24; // rcx
  __int64 v25; // r11
  _QWORD *v26; // rcx
  _QWORD *v27; // rdx
  _QWORD *v29; // rdx
  __int64 v30; // rdx
  _QWORD *v31; // rax
  int v32; // ecx
  unsigned int v33; // esi
  int v34; // eax
  _QWORD *v35; // rdx
  int v36; // eax
  int v37; // r8d
  __int64 v39; // [rsp+10h] [rbp-70h]
  int v40; // [rsp+1Ch] [rbp-64h]
  _QWORD *v41; // [rsp+28h] [rbp-58h] BYREF
  _QWORD *v42; // [rsp+30h] [rbp-50h] BYREF
  _QWORD *v43; // [rsp+38h] [rbp-48h] BYREF
  _QWORD *v44; // [rsp+40h] [rbp-40h] BYREF
  __int64 v45; // [rsp+48h] [rbp-38h]

  v3 = a3;
  v4 = a1;
  v5 = (__int64)a3;
  v7 = *(_QWORD **)a1;
  v41 = 0;
  if ( v7 == (_QWORD *)v5 || !v3 )
    goto LABEL_10;
  do
  {
    v3 = (_QWORD *)*v3;
    if ( v7 == v3 )
    {
      v8 = (__int64)a3;
      v41 = a3;
      for ( i = *a3; v7 != i; i = (_QWORD *)*i )
      {
        v41 = i;
        v8 = (__int64)i;
      }
      v45 = (__int64)v7;
      v44 = (_QWORD *)v8;
      if ( (unsigned __int8)sub_D4C320(a1 + 80, (__int64 *)&v44, &v42) )
      {
        v5 = v42[1];
        goto LABEL_10;
      }
      v33 = *(_DWORD *)(a1 + 104);
      v34 = *(_DWORD *)(a1 + 96);
      v35 = v42;
      ++*(_QWORD *)(a1 + 80);
      v36 = v34 + 1;
      v43 = v35;
      if ( 4 * v36 >= 3 * v33 )
      {
        v33 *= 2;
      }
      else if ( v33 - *(_DWORD *)(a1 + 100) - v36 > v33 >> 3 )
      {
LABEL_61:
        *(_DWORD *)(a1 + 96) = v36;
        if ( *v35 != -4096 )
          --*(_DWORD *)(a1 + 100);
        *v35 = v44;
        v5 = v45;
        v35[1] = v45;
LABEL_10:
        v10 = sub_986580(a2);
        if ( !v10 )
          goto LABEL_57;
LABEL_11:
        if ( !(unsigned int)sub_B46E30(v10) )
          v5 = 0;
        v11 = sub_986580(a2);
        v12 = v11;
        if ( !v11 )
          goto LABEL_58;
        goto LABEL_14;
      }
      sub_D4EA50(a1 + 80, v33);
      sub_D4C320(a1 + 80, (__int64 *)&v44, &v43);
      v35 = v43;
      v36 = *(_DWORD *)(a1 + 96) + 1;
      goto LABEL_61;
    }
  }
  while ( v3 );
  v5 = (__int64)a3;
  v10 = sub_986580(a2);
  if ( v10 )
    goto LABEL_11;
LABEL_57:
  v5 = 0;
  v11 = sub_986580(a2);
  v12 = v11;
  if ( v11 )
  {
LABEL_14:
    v40 = sub_B46E30(v11);
    v39 = (__int64)v41;
    if ( !v40 )
      goto LABEL_28;
    v13 = a2;
    v14 = v4;
    v15 = 0;
    v16 = v13;
    while ( 1 )
    {
      v17 = sub_B46EC0(v12, v15);
      if ( v16 != v17 )
      {
        v18 = *(_QWORD *)(v14 + 8);
        v19 = *(_DWORD *)(v18 + 24);
        v20 = *(_QWORD *)(v18 + 8);
        v21 = *(_QWORD **)v14;
        if ( !v19 )
        {
          v44 = 0;
          if ( v21 )
          {
LABEL_33:
            if ( (_QWORD *)v5 == v21 || !v5 )
              v5 = 0;
            goto LABEL_26;
          }
LABEL_51:
          *(_BYTE *)(v14 + 112) = 1;
          goto LABEL_26;
        }
        v22 = v19 - 1;
        v23 = v22 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v24 = (__int64 *)(v20 + 16LL * v23);
        v25 = *v24;
        if ( v17 == *v24 )
        {
LABEL_19:
          v26 = (_QWORD *)v24[1];
        }
        else
        {
          v32 = 1;
          while ( v25 != -4096 )
          {
            v37 = v32 + 1;
            v23 = v22 & (v32 + v23);
            v24 = (__int64 *)(v20 + 16LL * v23);
            v25 = *v24;
            if ( v17 == *v24 )
              goto LABEL_19;
            v32 = v37;
          }
          v26 = 0;
        }
        v44 = v26;
        if ( v21 == v26 )
          goto LABEL_51;
        if ( !v26 )
          goto LABEL_33;
        v27 = v26;
        while ( 1 )
        {
          v27 = (_QWORD *)*v27;
          if ( v21 == v27 )
            break;
          if ( !v27 )
            goto LABEL_36;
        }
        if ( !v39 )
        {
          v26 = (_QWORD *)*sub_D4EC30(v14 + 80, (__int64 *)&v44);
          v21 = *(_QWORD **)v14;
          v39 = (__int64)v41;
          v44 = v26;
          if ( v26 != v21 )
          {
            if ( !v26 )
              goto LABEL_33;
LABEL_36:
            if ( v26 == v21 )
            {
LABEL_41:
              v30 = (__int64)v44;
            }
            else
            {
              if ( v21 )
              {
                v29 = v21;
                do
                {
                  v29 = (_QWORD *)*v29;
                  if ( v26 == v29 )
                    goto LABEL_41;
                }
                while ( v29 );
              }
              v30 = *v26;
              v44 = (_QWORD *)*v26;
            }
            if ( v21 == (_QWORD *)v5 || !v5 )
              goto LABEL_49;
            if ( v5 != v30 && v30 )
            {
              v31 = (_QWORD *)v30;
              while ( 1 )
              {
                v31 = (_QWORD *)*v31;
                if ( v31 == (_QWORD *)v5 )
                  break;
                if ( !v31 )
                  goto LABEL_26;
              }
LABEL_49:
              v5 = v30;
            }
          }
        }
      }
LABEL_26:
      if ( v40 == ++v15 )
      {
        v4 = v14;
        goto LABEL_28;
      }
    }
  }
LABEL_58:
  v39 = (__int64)v41;
LABEL_28:
  if ( v39 )
  {
    *sub_D4EC30(v4 + 80, (__int64 *)&v41) = v5;
    return a3;
  }
  return (_QWORD **)v5;
}
