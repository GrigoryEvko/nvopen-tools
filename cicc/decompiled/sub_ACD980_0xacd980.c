// Function: sub_ACD980
// Address: 0xacd980
//
__int64 __fastcall sub_ACD980(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  unsigned int v5; // eax
  __int64 *v6; // rbx
  __int64 result; // rax
  unsigned int v8; // edx
  __int64 v9; // rbx
  int v10; // eax
  int v11; // ecx
  int v12; // eax
  bool v13; // cc
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // r12
  __int64 v19; // rdi
  __int64 v20; // rbx
  unsigned __int64 v21; // rax
  unsigned int v22; // eax
  __int64 v23; // rax
  unsigned __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rbx
  __int64 i; // rdx
  __int64 v28; // rcx
  __int64 j; // rdx
  __int64 v30; // rcx
  __int64 v31; // [rsp+0h] [rbp-A0h]
  unsigned int v32; // [rsp+8h] [rbp-98h]
  __int64 v33; // [rsp+8h] [rbp-98h]
  unsigned int v34; // [rsp+8h] [rbp-98h]
  __int64 v36; // [rsp+10h] [rbp-90h]
  __int64 v37; // [rsp+20h] [rbp-80h] BYREF
  __int64 v38; // [rsp+28h] [rbp-78h] BYREF
  __int64 v39; // [rsp+30h] [rbp-70h] BYREF
  __int64 v40; // [rsp+38h] [rbp-68h] BYREF
  unsigned int v41; // [rsp+40h] [rbp-60h]
  __int64 v42; // [rsp+50h] [rbp-50h]

  v4 = *a1;
  v5 = *(_DWORD *)(a3 + 8);
  v39 = a2;
  v41 = v5;
  if ( v5 > 0x40 )
    sub_C43780(&v40, a3);
  else
    v40 = *(_QWORD *)a3;
  if ( (unsigned __int8)sub_AC6600(v4 + 304, (int *)&v39, &v37) )
  {
    v6 = (__int64 *)(v37 + 24);
    if ( v41 > 0x40 && v40 )
      j_j___libc_free_0_0(v40);
    result = *v6;
    if ( *v6 )
      return result;
LABEL_16:
    v16 = sub_BCCE00(a1, *(unsigned int *)(a3 + 8));
    v17 = sub_BCE1B0(v16, a2);
    result = sub_BD2C40(40, unk_3F289A4);
    if ( result )
    {
      v36 = result;
      sub_AC2FF0(result, v17, a3);
      result = v36;
    }
    v18 = *v6;
    *v6 = result;
    if ( v18 )
    {
      if ( *(_DWORD *)(v18 + 32) > 0x40u )
      {
        v19 = *(_QWORD *)(v18 + 24);
        if ( v19 )
          j_j___libc_free_0_0(v19);
      }
      sub_BD7260(v18);
      sub_BD2DD0(v18);
      return *v6;
    }
    return result;
  }
  v8 = *(_DWORD *)(v4 + 328);
  v9 = v37;
  v10 = *(_DWORD *)(v4 + 320);
  ++*(_QWORD *)(v4 + 304);
  v38 = v9;
  v12 = v10 + 1;
  if ( 4 * v12 >= 3 * v8 )
  {
    v20 = *(_QWORD *)(v4 + 312);
    v32 = v8;
    v11 = 2 * v8;
    v21 = (((((((((unsigned int)(v11 - 1) | ((unsigned __int64)(unsigned int)(v11 - 1) >> 1)) >> 2)
              | (unsigned int)(v11 - 1)
              | ((unsigned __int64)(unsigned int)(v11 - 1) >> 1)) >> 4)
            | (((unsigned int)(v11 - 1) | ((unsigned __int64)(unsigned int)(v11 - 1) >> 1)) >> 2)
            | (unsigned int)(v11 - 1)
            | ((unsigned __int64)(unsigned int)(v11 - 1) >> 1)) >> 8)
          | (((((unsigned int)(v11 - 1) | ((unsigned __int64)(unsigned int)(v11 - 1) >> 1)) >> 2)
            | (unsigned int)(v11 - 1)
            | ((unsigned __int64)(unsigned int)(v11 - 1) >> 1)) >> 4)
          | (((unsigned int)(v11 - 1) | ((unsigned __int64)(unsigned int)(v11 - 1) >> 1)) >> 2)
          | (unsigned int)(v11 - 1)
          | ((unsigned __int64)(unsigned int)(v11 - 1) >> 1)) >> 16)
        | (((((((unsigned int)(v11 - 1) | ((unsigned __int64)(unsigned int)(v11 - 1) >> 1)) >> 2)
            | (unsigned int)(v11 - 1)
            | ((unsigned __int64)(unsigned int)(v11 - 1) >> 1)) >> 4)
          | (((unsigned int)(v11 - 1) | ((unsigned __int64)(unsigned int)(v11 - 1) >> 1)) >> 2)
          | (unsigned int)(v11 - 1)
          | ((unsigned __int64)(unsigned int)(v11 - 1) >> 1)) >> 8)
        | (((((unsigned int)(v11 - 1) | ((unsigned __int64)(unsigned int)(v11 - 1) >> 1)) >> 2)
          | (unsigned int)(v11 - 1)
          | ((unsigned __int64)(unsigned int)(v11 - 1) >> 1)) >> 4)
        | (((unsigned int)(v11 - 1) | ((unsigned __int64)(unsigned int)(v11 - 1) >> 1)) >> 2)
        | (unsigned int)(v11 - 1)
        | ((unsigned __int64)(unsigned int)(v11 - 1) >> 1);
    v22 = v21 + 1;
    if ( v22 < 0x40 )
      v22 = 64;
    *(_DWORD *)(v4 + 328) = v22;
    v23 = sub_C7D670(32LL * v22, 8);
    *(_QWORD *)(v4 + 312) = v23;
    if ( v20 )
    {
      v33 = 32LL * v32;
      sub_ACD780(v4 + 304, v20, v20 + v33);
      sub_C7D6A0(v20, v33, 8);
    }
    else
    {
      *(_QWORD *)(v4 + 320) = 0;
      LODWORD(v42) = -1;
      BYTE4(v42) = 1;
      for ( i = v23 + 32LL * *(unsigned int *)(v4 + 328); i != v23; v23 += 32 )
      {
        if ( v23 )
        {
          v28 = v42;
          *(_DWORD *)(v23 + 16) = 0;
          *(_QWORD *)(v23 + 8) = -1;
          *(_QWORD *)v23 = v28;
        }
      }
    }
    goto LABEL_32;
  }
  if ( v8 - *(_DWORD *)(v4 + 324) - v12 <= v8 >> 3 )
  {
    v34 = v8;
    v31 = *(_QWORD *)(v4 + 312);
    v24 = ((((((((((v8 - 1) | ((unsigned __int64)(v8 - 1) >> 1)) >> 2) | (v8 - 1) | ((unsigned __int64)(v8 - 1) >> 1)) >> 4)
             | (((v8 - 1) | ((unsigned __int64)(v8 - 1) >> 1)) >> 2)
             | (v8 - 1)
             | ((unsigned __int64)(v8 - 1) >> 1)) >> 8)
           | (((((v8 - 1) | ((unsigned __int64)(v8 - 1) >> 1)) >> 2) | (v8 - 1) | ((unsigned __int64)(v8 - 1) >> 1)) >> 4)
           | (((v8 - 1) | ((unsigned __int64)(v8 - 1) >> 1)) >> 2)
           | (v8 - 1)
           | ((unsigned __int64)(v8 - 1) >> 1)) >> 16)
         | (((((((v8 - 1) | ((unsigned __int64)(v8 - 1) >> 1)) >> 2) | (v8 - 1) | ((unsigned __int64)(v8 - 1) >> 1)) >> 4)
           | (((v8 - 1) | ((unsigned __int64)(v8 - 1) >> 1)) >> 2)
           | (v8 - 1)
           | ((unsigned __int64)(v8 - 1) >> 1)) >> 8)
         | (((((v8 - 1) | ((unsigned __int64)(v8 - 1) >> 1)) >> 2) | (v8 - 1) | ((unsigned __int64)(v8 - 1) >> 1)) >> 4)
         | (((v8 - 1) | ((unsigned __int64)(v8 - 1) >> 1)) >> 2)
         | (v8 - 1)
         | ((unsigned __int64)(v8 - 1) >> 1))
        + 1;
    if ( (unsigned int)v24 < 0x40 )
      LODWORD(v24) = 64;
    *(_DWORD *)(v4 + 328) = v24;
    v25 = sub_C7D670(32LL * (unsigned int)v24, 8);
    *(_QWORD *)(v4 + 312) = v25;
    if ( v31 )
    {
      v26 = 32LL * v34;
      sub_ACD780(v4 + 304, v31, v31 + v26);
      sub_C7D6A0(v31, v26, 8);
    }
    else
    {
      *(_QWORD *)(v4 + 320) = 0;
      LODWORD(v42) = -1;
      BYTE4(v42) = 1;
      for ( j = v25 + 32LL * *(unsigned int *)(v4 + 328); j != v25; v25 += 32 )
      {
        if ( v25 )
        {
          v30 = v42;
          *(_DWORD *)(v25 + 16) = 0;
          *(_QWORD *)(v25 + 8) = -1;
          *(_QWORD *)v25 = v30;
        }
      }
    }
LABEL_32:
    sub_AC6600(v4 + 304, (int *)&v39, &v38);
    v9 = v38;
    v12 = *(_DWORD *)(v4 + 320) + 1;
  }
  *(_DWORD *)(v4 + 320) = v12;
  if ( *(_DWORD *)v9 == -1 && *(_BYTE *)(v9 + 4) && !*(_DWORD *)(v9 + 16) && *(_QWORD *)(v9 + 8) == -1 )
  {
    *(_DWORD *)v9 = v39;
    *(_BYTE *)(v9 + 4) = BYTE4(v39);
  }
  else
  {
    --*(_DWORD *)(v4 + 324);
    v13 = *(_DWORD *)(v9 + 16) <= 0x40u;
    *(_DWORD *)v9 = v39;
    *(_BYTE *)(v9 + 4) = BYTE4(v39);
    if ( !v13 )
    {
      v14 = *(_QWORD *)(v9 + 8);
      if ( v14 )
        j_j___libc_free_0_0(v14);
    }
  }
  v15 = v40;
  *(_QWORD *)(v9 + 24) = 0;
  v6 = (__int64 *)(v9 + 24);
  *(v6 - 2) = v15;
  *((_DWORD *)v6 - 2) = v41;
  result = *v6;
  if ( !*v6 )
    goto LABEL_16;
  return result;
}
