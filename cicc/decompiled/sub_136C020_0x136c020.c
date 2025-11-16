// Function: sub_136C020
// Address: 0x136c020
//
__int64 __fastcall sub_136C020(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rdi
  int v8; // eax
  int v9; // edx
  int v10; // esi
  __int64 v11; // rcx
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 *v16; // rax
  __int64 v17; // rdx
  __int64 *v18; // r14
  __int64 *v19; // r15
  __int64 result; // rax
  __int64 *v21; // rbx
  __int64 v22; // r14
  __int64 v23; // rdi
  int v24; // eax
  int v25; // ecx
  int v26; // ecx
  __int64 v27; // r10
  unsigned int v28; // edx
  __int64 *v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rax
  __int64 v32; // r11
  __int64 v33; // r10
  int v34; // eax
  __int64 *v35; // rax
  int v36; // eax
  int v37; // r8d
  int v38; // eax
  int v39; // r9d
  __int64 *v42; // [rsp+10h] [rbp-A0h]
  __int64 v43; // [rsp+18h] [rbp-98h]
  unsigned int v44; // [rsp+34h] [rbp-7Ch]
  __int64 v46; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v47; // [rsp+48h] [rbp-68h]
  __int64 v48; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v49; // [rsp+58h] [rbp-58h]
  _QWORD *v50; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v51; // [rsp+68h] [rbp-48h]
  _QWORD *v52; // [rsp+70h] [rbp-40h] BYREF
  unsigned int v53; // [rsp+78h] [rbp-38h]

  v47 = 128;
  sub_16A4EF0(&v46, a3, 0);
  v7 = *a1;
  v8 = -1;
  v9 = *(_DWORD *)(*a1 + 184);
  if ( v9 )
  {
    v10 = v9 - 1;
    v11 = *(_QWORD *)(v7 + 168);
    v12 = (v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v13 = (__int64 *)(v11 + 16LL * v12);
    v14 = *v13;
    if ( a2 == *v13 )
    {
LABEL_3:
      v8 = *((_DWORD *)v13 + 2);
    }
    else
    {
      v38 = 1;
      while ( v14 != -8 )
      {
        v39 = v38 + 1;
        v12 = v10 & (v38 + v12);
        v13 = (__int64 *)(v11 + 16LL * v12);
        v14 = *v13;
        if ( a2 == *v13 )
          goto LABEL_3;
        v38 = v39;
      }
      v8 = -1;
    }
  }
  LODWORD(v52) = v8;
  v15 = sub_1370CD0(v7, &v52);
  v49 = 128;
  sub_16A4EF0(&v48, v15, 0);
  v51 = 128;
  sub_16A4EF0(&v50, 0, 0);
  v16 = *(__int64 **)(a4 + 16);
  if ( v16 == *(__int64 **)(a4 + 8) )
    v17 = *(unsigned int *)(a4 + 28);
  else
    v17 = *(unsigned int *)(a4 + 24);
  v18 = &v16[v17];
  if ( v16 != v18 )
  {
    while ( 1 )
    {
      v19 = v16;
      if ( (unsigned __int64)*v16 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v18 == ++v16 )
        goto LABEL_9;
    }
    if ( v18 != v16 )
    {
      v21 = v18;
      v22 = *v16;
      do
      {
        v23 = *a1;
        v24 = -1;
        v25 = *(_DWORD *)(*a1 + 184);
        if ( v25 )
        {
          v26 = v25 - 1;
          v27 = *(_QWORD *)(v23 + 168);
          v28 = v26 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
          v29 = (__int64 *)(v27 + 16LL * v28);
          v30 = *v29;
          if ( *v29 == v22 )
          {
LABEL_23:
            v24 = *((_DWORD *)v29 + 2);
          }
          else
          {
            v36 = 1;
            while ( v30 != -8 )
            {
              v37 = v36 + 1;
              v28 = v26 & (v36 + v28);
              v29 = (__int64 *)(v27 + 16LL * v28);
              v30 = *v29;
              if ( *v29 == v22 )
                goto LABEL_23;
              v36 = v37;
            }
            v24 = -1;
          }
        }
        LODWORD(v52) = v24;
        v31 = sub_1370CD0(v23, &v52);
        if ( v51 > 0x40 )
        {
          *v50 = v31;
          memset(v50 + 1, 0, 8 * (unsigned int)(((unsigned __int64)v51 + 63) >> 6) - 8);
        }
        else
        {
          v50 = (_QWORD *)(v31 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v51));
        }
        sub_16A7C10(&v50, &v46);
        sub_16A9D70(&v52, &v50, &v48);
        if ( v51 > 0x40 && v50 )
          j_j___libc_free_0_0(v50);
        v32 = (__int64)v52;
        v50 = v52;
        v51 = v53;
        v33 = *a1;
        v44 = v53;
        if ( v53 > 0x40 )
        {
          v42 = v52;
          v43 = *a1;
          v34 = sub_16A57B0(&v50);
          v33 = v43;
          v32 = -1;
          if ( v44 - v34 <= 0x40 )
            v32 = *v42;
        }
        sub_136BC40(v33, v22, v32);
        v35 = v19 + 1;
        if ( v19 + 1 == v21 )
          break;
        v22 = *v35;
        for ( ++v19; (unsigned __int64)*v35 >= 0xFFFFFFFFFFFFFFFELL; v19 = v35 )
        {
          if ( v21 == ++v35 )
            goto LABEL_9;
          v22 = *v35;
        }
      }
      while ( v21 != v19 );
    }
  }
LABEL_9:
  result = sub_136BC40(*a1, a2, a3);
  if ( v51 > 0x40 && v50 )
    result = j_j___libc_free_0_0(v50);
  if ( v49 > 0x40 && v48 )
    result = j_j___libc_free_0_0(v48);
  if ( v47 > 0x40 )
  {
    if ( v46 )
      return j_j___libc_free_0_0(v46);
  }
  return result;
}
