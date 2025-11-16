// Function: sub_2BF9BD0
// Address: 0x2bf9bd0
//
__int64 __fastcall sub_2BF9BD0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rax
  int v8; // r13d
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 *v11; // rbx
  __int64 *v12; // r12
  _QWORD *v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  int v18; // r15d
  _QWORD *v19; // r11
  __int64 v20; // r9
  int v21; // edx
  _QWORD *v22; // r8
  __int64 v23; // rdi
  int v24; // eax
  __int64 v25; // r15
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  __int64 v29; // rdx
  __int64 v30; // rdi
  int v31; // r10d
  int v32; // r10d
  __int64 v33; // rdx
  __int64 v34; // rdi
  __int64 v35; // [rsp+28h] [rbp-A8h] BYREF
  __int64 v36; // [rsp+30h] [rbp-A0h] BYREF
  _QWORD *v37; // [rsp+38h] [rbp-98h]
  __int64 v38; // [rsp+40h] [rbp-90h]
  __int64 v39; // [rsp+48h] [rbp-88h]
  _QWORD *v40; // [rsp+50h] [rbp-80h] BYREF
  __int64 v41; // [rsp+58h] [rbp-78h]
  _QWORD v42[14]; // [rsp+60h] [rbp-70h] BYREF

  v1 = *(_QWORD *)(a1 + 48);
  v35 = a1;
  if ( v1 )
  {
    do
    {
      v2 = v1;
      v1 = *(_QWORD *)(v1 + 48);
    }
    while ( v1 );
    v35 = v2;
  }
  v36 = 0;
  v40 = v42;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v41 = 0x800000000LL;
  if ( v42 != sub_2BEF2F0(v42, (__int64)v42, &v35) || (sub_2BF9A30((__int64)&v36, v35, v3, v4, v5, v6), !(_DWORD)v41) )
LABEL_56:
    BUG();
  v7 = 0;
  v8 = 0;
LABEL_7:
  v9 = v40[v7];
  v10 = *(unsigned int *)(v9 + 64);
  if ( *(_DWORD *)(v9 + 64) )
  {
    v11 = *(__int64 **)(v9 + 56);
    v12 = &v11[v10];
    while ( (_DWORD)v38 )
    {
      if ( !(_DWORD)v39 )
      {
        ++v36;
        goto LABEL_31;
      }
      v18 = 1;
      v19 = 0;
      v20 = (__int64)v37;
      v21 = (v39 - 1) & (((unsigned int)*v11 >> 9) ^ ((unsigned int)*v11 >> 4));
      v22 = &v37[v21];
      v23 = *v22;
      if ( *v22 == *v11 )
      {
LABEL_11:
        if ( v12 == ++v11 )
          goto LABEL_25;
      }
      else
      {
        while ( v23 != -4096 )
        {
          if ( v19 || v23 != -8192 )
            v22 = v19;
          v21 = (v39 - 1) & (v18 + v21);
          v23 = v37[v21];
          if ( *v11 == v23 )
            goto LABEL_11;
          ++v18;
          v19 = v22;
          v22 = &v37[v21];
        }
        if ( !v19 )
          v19 = v22;
        v24 = v38 + 1;
        ++v36;
        if ( 4 * ((int)v38 + 1) < (unsigned int)(3 * v39) )
        {
          if ( (int)v39 - HIDWORD(v38) - v24 > (unsigned int)v39 >> 3 )
            goto LABEL_20;
          sub_2BF9860((__int64)&v36, v39);
          if ( !(_DWORD)v39 )
          {
LABEL_57:
            LODWORD(v38) = v38 + 1;
            BUG();
          }
          v22 = v37;
          v20 = 0;
          v32 = 1;
          LODWORD(v33) = (v39 - 1) & (((unsigned int)*v11 >> 9) ^ ((unsigned int)*v11 >> 4));
          v19 = &v37[(unsigned int)v33];
          v34 = *v19;
          v24 = v38 + 1;
          if ( *v11 == *v19 )
            goto LABEL_20;
          while ( v34 != -4096 )
          {
            if ( v34 == -8192 && !v20 )
              v20 = (__int64)v19;
            v33 = ((_DWORD)v39 - 1) & (unsigned int)(v33 + v32);
            v19 = &v37[v33];
            v34 = *v19;
            if ( *v11 == *v19 )
              goto LABEL_20;
            ++v32;
          }
          goto LABEL_43;
        }
LABEL_31:
        sub_2BF9860((__int64)&v36, 2 * v39);
        if ( !(_DWORD)v39 )
          goto LABEL_57;
        v22 = v37;
        LODWORD(v29) = (v39 - 1) & (((unsigned int)*v11 >> 9) ^ ((unsigned int)*v11 >> 4));
        v19 = &v37[(unsigned int)v29];
        v30 = *v19;
        v24 = v38 + 1;
        if ( *v19 == *v11 )
          goto LABEL_20;
        v31 = 1;
        v20 = 0;
        while ( v30 != -4096 )
        {
          if ( v30 == -8192 && !v20 )
            v20 = (__int64)v19;
          v29 = ((_DWORD)v39 - 1) & (unsigned int)(v29 + v31);
          v19 = &v37[v29];
          v30 = *v19;
          if ( *v11 == *v19 )
            goto LABEL_20;
          ++v31;
        }
LABEL_43:
        if ( v20 )
          v19 = (_QWORD *)v20;
LABEL_20:
        LODWORD(v38) = v24;
        if ( *v19 != -4096 )
          --HIDWORD(v38);
        v25 = *v11;
        *v19 = *v11;
        v26 = (unsigned int)v41;
        v27 = (unsigned int)v41 + 1LL;
        if ( v27 > HIDWORD(v41) )
        {
          sub_C8D5F0((__int64)&v40, v42, v27, 8u, (__int64)v22, v20);
          v26 = (unsigned int)v41;
        }
        ++v11;
        v40[v26] = v25;
        LODWORD(v41) = v41 + 1;
        if ( v12 == v11 )
        {
LABEL_25:
          v7 = (unsigned int)(v8 + 1);
          v8 = v7;
          if ( (unsigned int)v41 <= (unsigned int)v7 )
            goto LABEL_56;
          goto LABEL_7;
        }
      }
    }
    v13 = &v40[(unsigned int)v41];
    if ( v13 == sub_2BEF2F0(v40, (__int64)v13, v11) )
      sub_2BF9A30((__int64)&v36, *v11, v14, v15, v16, v17);
    goto LABEL_11;
  }
  if ( v40 != v42 )
    _libc_free((unsigned __int64)v40);
  sub_C7D6A0((__int64)v37, 8LL * (unsigned int)v39, 8);
  return *(_QWORD *)(v9 + 104);
}
