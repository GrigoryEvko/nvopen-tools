// Function: sub_ACE4D0
// Address: 0xace4d0
//
__int64 __fastcall sub_ACE4D0(__int64 *a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rdi
  char v9; // al
  __int64 *v10; // r8
  __int64 *v11; // r12
  __int64 result; // rax
  __int64 *v13; // rsi
  int v14; // eax
  int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // r15
  __int64 *v20; // r12
  __int64 v21; // rax
  _QWORD *i; // rbx
  __int64 v23; // rax
  __int64 v24; // r14
  __int64 v25; // r13
  __int64 v26; // rdi
  __int64 v27; // rax
  __int64 v28; // rdi
  char v29; // al
  __int64 v30; // rsi
  __int64 v31; // rdi
  __int64 v33; // [rsp+10h] [rbp-B0h]
  __int64 v35; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v36; // [rsp+28h] [rbp-98h] BYREF
  _QWORD v37[4]; // [rsp+30h] [rbp-90h] BYREF
  __int64 v38; // [rsp+50h] [rbp-70h] BYREF
  __int64 v39; // [rsp+58h] [rbp-68h] BYREF
  _QWORD *v40; // [rsp+60h] [rbp-60h]
  int v41; // [rsp+70h] [rbp-50h]
  char v42; // [rsp+74h] [rbp-4Ch]
  _QWORD v43[9]; // [rsp+78h] [rbp-48h] BYREF

  v6 = *a1;
  v38 = a2;
  v7 = sub_C33340(a1, a2, a3, a4, a5);
  if ( *a3 == v7 )
    sub_C3C790(&v39, a3);
  else
    sub_C33EB0(&v39, a3);
  v8 = v6 + 368;
  v9 = sub_AC6AC0(v6 + 368, (int *)&v38, &v35);
  v10 = &v38;
  if ( !v9 )
  {
    v13 = (__int64 *)*(unsigned int *)(v6 + 392);
    v36 = v35;
    v14 = *(_DWORD *)(v6 + 384);
    ++*(_QWORD *)(v6 + 368);
    v15 = v14 + 1;
    if ( 4 * v15 >= (unsigned int)(3 * (_DWORD)v13) )
    {
      LODWORD(v13) = 2 * (_DWORD)v13;
    }
    else
    {
      v16 = (unsigned int)((_DWORD)v13 - *(_DWORD *)(v6 + 388) - v15);
      v17 = (unsigned int)v13 >> 3;
      if ( (unsigned int)v16 > (unsigned int)v17 )
        goto LABEL_10;
    }
    sub_ACE320(v6 + 368, (int)v13);
    v8 = v6 + 368;
    v13 = &v38;
    sub_AC6AC0(v6 + 368, (int *)&v38, &v36);
    v15 = *(_DWORD *)(v6 + 384) + 1;
LABEL_10:
    *(_DWORD *)(v6 + 384) = v15;
    v18 = sub_C33690(v8, v13, v16, v17, v10);
    if ( v7 == v18 )
      sub_C3C5A0(v37, v7, 1);
    else
      sub_C36740(v37, v18, 1);
    v42 = 1;
    v41 = -1;
    if ( v7 == v37[0] )
      sub_C3C840(v43, v37);
    else
      sub_C338E0(v43, v37);
    sub_91D830(v37);
    v19 = v36;
    if ( *(_DWORD *)v36 != v41
      || *(_BYTE *)(v36 + 4) != v42
      || (v27 = *(_QWORD *)(v36 + 8), v27 != v43[0])
      || ((v28 = v36 + 8, v7 == v27) ? (v29 = sub_C3E590(v28)) : (v29 = sub_C33D00(v28)), v19 = v36, !v29) )
    {
      --*(_DWORD *)(v6 + 388);
    }
    v20 = (__int64 *)(v19 + 8);
    sub_91D830(v43);
    *(_DWORD *)v19 = v38;
    *(_BYTE *)(v19 + 4) = BYTE4(v38);
    if ( v7 == *(_QWORD *)(v19 + 8) )
    {
      v21 = v39;
      if ( v7 == v39 )
      {
        if ( v20 != &v39 )
        {
          v30 = *(_QWORD *)(v19 + 16);
          if ( v30 )
          {
            v31 = v30 + 24LL * *(_QWORD *)(v30 - 8);
            if ( v30 != v31 )
            {
              do
              {
                sub_91D830((_QWORD *)(v31 - 24));
                v31 -= 24;
              }
              while ( *(_QWORD *)(v19 + 16) != v31 );
            }
            j_j_j___libc_free_0_0(v31 - 8);
          }
          sub_C3C840(v19 + 8, &v39);
          v21 = v39;
        }
        goto LABEL_19;
      }
    }
    else if ( v7 != v39 )
    {
      sub_C33870(v19 + 8, &v39);
      v21 = v39;
      goto LABEL_19;
    }
    if ( v20 != &v39 )
    {
      sub_91D830((_QWORD *)(v19 + 8));
      v26 = v19 + 8;
      if ( v7 != v39 )
      {
        sub_C338E0(v26, &v39);
        v21 = v39;
        goto LABEL_19;
      }
      sub_C3C840(v26, &v39);
    }
    v21 = v39;
LABEL_19:
    *(_QWORD *)(v19 + 32) = 0;
    v11 = (__int64 *)(v19 + 32);
    if ( v7 != v21 )
      goto LABEL_5;
    goto LABEL_20;
  }
  v11 = (__int64 *)(v35 + 32);
  if ( v7 != v39 )
  {
LABEL_5:
    sub_C338F0(&v39);
    goto LABEL_6;
  }
LABEL_20:
  if ( !v40 )
  {
LABEL_6:
    result = *v11;
    if ( *v11 )
      return result;
    goto LABEL_24;
  }
  for ( i = &v40[3 * *(v40 - 1)]; v40 != i; sub_91D830(i) )
    i -= 3;
  j_j_j___libc_free_0_0(i - 1);
  result = *v11;
  if ( !*v11 )
  {
LABEL_24:
    v23 = sub_BCB1D0(a1, *a3);
    v24 = sub_BCE1B0(v23, a2);
    result = sub_BD2C40(48, unk_3F289A4);
    if ( result )
    {
      v33 = result;
      sub_AC3040(result, v24, a3);
      result = v33;
    }
    v25 = *v11;
    *v11 = result;
    if ( v25 )
    {
      sub_91D830((_QWORD *)(v25 + 24));
      sub_BD7260(v25);
      sub_BD2DD0(v25);
      return *v11;
    }
  }
  return result;
}
