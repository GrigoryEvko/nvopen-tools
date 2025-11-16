// Function: sub_1E07970
// Address: 0x1e07970
//
__int64 __fastcall sub_1E07970(__int64 *a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r14
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 *v9; // rax
  __int64 result; // rax
  __int64 *v11; // r12
  __int64 *v12; // rbx
  __int64 *v13; // r13
  __int64 v14; // r10
  unsigned int v15; // esi
  __int64 v16; // rdi
  __int64 v17; // r9
  unsigned int v18; // eax
  int v19; // eax
  __int64 v20; // r12
  int v21; // edx
  __int64 v22; // rdi
  int v23; // r9d
  unsigned int v24; // eax
  __int64 v25; // r8
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned int v28; // r8d
  int v29; // esi
  __int64 v30; // rdx
  int v31; // edi
  __int64 v32; // r15
  __int64 *v33; // r14
  unsigned int v34; // r12d
  int v35; // ecx
  int v36; // ecx
  __int64 *v37; // r12
  unsigned int v38; // [rsp+10h] [rbp-90h]
  __int64 v40; // [rsp+28h] [rbp-78h]
  __int64 v41; // [rsp+30h] [rbp-70h]
  _QWORD *v43; // [rsp+38h] [rbp-68h]
  __int64 *v44; // [rsp+40h] [rbp-60h]
  unsigned int v45; // [rsp+48h] [rbp-58h]
  __int64 v46; // [rsp+58h] [rbp-48h] BYREF
  __int64 v47; // [rsp+60h] [rbp-40h] BYREF
  __int64 v48[7]; // [rsp+68h] [rbp-38h] BYREF

  v4 = *a1;
  v5 = (a1[1] - *a1) >> 3;
  v38 = v5;
  if ( (unsigned int)v5 > 1 )
  {
    v6 = 8;
    v7 = 8LL * (unsigned int)v5;
    while ( 1 )
    {
      v8 = *(_QWORD *)(v4 + v6);
      v6 += 8;
      v48[0] = v8;
      v9 = sub_1E071E0((__int64)(a1 + 3), v48);
      v9[4] = *(_QWORD *)(*a1 + 8LL * *((unsigned int *)v9 + 3));
      if ( v7 == v6 )
        break;
      v4 = *a1;
    }
  }
  result = v38 - 1;
  if ( (unsigned int)result > 1 )
  {
    v41 = 8 * result;
    v40 = (__int64)(a1 + 3);
    v45 = v38;
    while ( 1 )
    {
      v46 = *(_QWORD *)(*a1 + v41);
      v11 = sub_1E071E0(v40, &v46);
      v12 = (__int64 *)v11[5];
      *((_DWORD *)v11 + 4) = *((_DWORD *)v11 + 3);
      v13 = &v12[*((unsigned int *)v11 + 12)];
      if ( v12 != v13 )
        break;
LABEL_29:
      result = --v45;
      v41 -= 8;
      if ( v45 == 2 )
      {
        if ( v38 > 2 )
        {
          v43 = a1;
          v32 = 16;
          do
          {
            v47 = *(_QWORD *)(*v43 + v32);
            v33 = sub_1E071E0(v40, &v47);
            v34 = *((_DWORD *)sub_1E071E0(v40, (__int64 *)(*v43 + 8LL * *((unsigned int *)v33 + 4))) + 2);
            v48[0] = v33[4];
            while ( *((_DWORD *)sub_1E071E0(v40, v48) + 2) > v34 )
              v48[0] = sub_1E071E0(v40, v48)[4];
            result = v48[0];
            v32 += 8;
            v33[4] = v48[0];
          }
          while ( v32 != 8LL * v38 );
        }
        return result;
      }
    }
    v44 = v11;
    while ( 1 )
    {
      v19 = *((_DWORD *)a1 + 12);
      if ( !v19 )
        goto LABEL_13;
      v20 = *v12;
      v21 = v19 - 1;
      v22 = a1[4];
      v23 = 1;
      v24 = (v19 - 1) & (((unsigned int)*v12 >> 9) ^ ((unsigned int)*v12 >> 4));
      v25 = *(_QWORD *)(v22 + 72LL * (v21 & (((unsigned int)*v12 >> 9) ^ ((unsigned int)*v12 >> 4))));
      if ( *v12 == v25 )
      {
LABEL_16:
        v26 = sub_1E05220(a2, *v12);
        if ( !v26 || a3 <= *(_DWORD *)(v26 + 16) )
        {
          v27 = sub_1E07330(a1, v20, v45);
          v28 = *((_DWORD *)a1 + 12);
          v47 = v27;
          if ( !v28 )
          {
            ++a1[3];
            goto LABEL_20;
          }
          v14 = a1[4];
          v15 = (v28 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
          v16 = v14 + 72LL * v15;
          v17 = *(_QWORD *)v16;
          if ( v27 == *(_QWORD *)v16 )
          {
            v18 = *(_DWORD *)(v16 + 16);
          }
          else
          {
            v35 = 1;
            v30 = 0;
            while ( v17 != -8 )
            {
              if ( v30 || v17 != -16 )
                v16 = v30;
              v15 = (v28 - 1) & (v35 + v15);
              v37 = (__int64 *)(v14 + 72LL * v15);
              v17 = *v37;
              if ( v27 == *v37 )
              {
                v18 = *((_DWORD *)v37 + 4);
                goto LABEL_11;
              }
              v30 = v16;
              ++v35;
              v16 = v14 + 72LL * v15;
            }
            v36 = *((_DWORD *)a1 + 10);
            if ( !v30 )
              v30 = v16;
            ++a1[3];
            v31 = v36 + 1;
            if ( 4 * (v36 + 1) >= 3 * v28 )
            {
LABEL_20:
              v29 = 2 * v28;
            }
            else
            {
              if ( v28 - *((_DWORD *)a1 + 11) - v31 > v28 >> 3 )
                goto LABEL_22;
              v29 = v28;
            }
            sub_1E06FA0(v40, v29);
            sub_1E06EF0(v40, &v47, v48);
            v30 = v48[0];
            v27 = v47;
            v31 = *((_DWORD *)a1 + 10) + 1;
LABEL_22:
            *((_DWORD *)a1 + 10) = v31;
            if ( *(_QWORD *)v30 != -8 )
              --*((_DWORD *)a1 + 11);
            *(_QWORD *)v30 = v27;
            *(_QWORD *)(v30 + 40) = v30 + 56;
            *(_QWORD *)(v30 + 48) = 0x200000000LL;
            v18 = 0;
            *(_OWORD *)(v30 + 8) = 0;
            *(_OWORD *)(v30 + 24) = 0;
            *(_OWORD *)(v30 + 56) = 0;
          }
LABEL_11:
          if ( *((_DWORD *)v44 + 4) > v18 )
            *((_DWORD *)v44 + 4) = v18;
        }
LABEL_13:
        if ( v13 == ++v12 )
          goto LABEL_29;
      }
      else
      {
        while ( v25 != -8 )
        {
          v24 = v21 & (v23 + v24);
          v25 = *(_QWORD *)(v22 + 72LL * v24);
          if ( v20 == v25 )
            goto LABEL_16;
          ++v23;
        }
        if ( v13 == ++v12 )
          goto LABEL_29;
      }
    }
  }
  return result;
}
