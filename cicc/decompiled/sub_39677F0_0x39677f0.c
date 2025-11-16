// Function: sub_39677F0
// Address: 0x39677f0
//
__int64 __fastcall sub_39677F0(_QWORD *a1, __int64 a2, _BYTE *a3)
{
  __int64 v5; // rax
  _QWORD *v6; // r13
  int v7; // esi
  __int64 v8; // rax
  _QWORD *v9; // rdi
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 *v12; // rax
  _QWORD *v13; // rdi
  int v14; // esi
  __int64 v15; // r9
  __int64 v16; // rax
  __int64 v17; // r8
  __int64 v18; // rcx
  __int64 v19; // rdx
  int *v20; // rax
  __int64 v22; // rbx
  __int64 v23; // rsi
  int v24; // r9d
  __int64 v25; // rcx
  int v26; // r11d
  unsigned int v27; // edx
  unsigned int v28; // r13d
  int *v29; // rax
  __int64 v30; // r8
  __int64 v31; // r10
  __int64 v32; // rax
  __int64 *v33; // rax
  int v34; // esi
  __int64 v35; // rax
  __int64 v36; // r9
  __int64 v37; // rcx
  __int64 v38; // rdx
  int v39; // r15d
  __int64 *v40; // r11
  int v41; // r15d
  int *v42; // r10
  int v43; // eax
  int v44; // edx
  __int64 v45; // rax
  __int64 v46[2]; // [rsp+8h] [rbp-58h] BYREF
  int v47; // [rsp+1Ch] [rbp-44h] BYREF
  __int64 v48; // [rsp+20h] [rbp-40h] BYREF
  int *v49[7]; // [rsp+28h] [rbp-38h] BYREF

  v5 = a1[2];
  v6 = (_QWORD *)*a1;
  v46[0] = a2;
  v7 = *((_DWORD *)sub_3967280(v5 + 264, v46) + 2);
  v8 = v6[2];
  if ( v8 )
  {
    v9 = v6 + 1;
    do
    {
      while ( 1 )
      {
        v10 = *(_QWORD *)(v8 + 16);
        v11 = *(_QWORD *)(v8 + 24);
        if ( v7 <= *(_DWORD *)(v8 + 32) )
          break;
        v8 = *(_QWORD *)(v8 + 24);
        if ( !v11 )
          goto LABEL_6;
      }
      v9 = (_QWORD *)v8;
      v8 = *(_QWORD *)(v8 + 16);
    }
    while ( v10 );
LABEL_6:
    if ( v6 + 1 != v9 && v7 >= *((_DWORD *)v9 + 8) )
    {
      *a3 = 1;
      v12 = sub_3967280(a1[2] + 264LL, v46);
      v13 = (_QWORD *)*a1;
      v14 = *((_DWORD *)v12 + 2);
      v15 = *a1 + 8LL;
      LODWORD(v48) = v14;
      v16 = v13[2];
      v17 = v15;
      if ( v16 )
      {
        do
        {
          while ( 1 )
          {
            v18 = *(_QWORD *)(v16 + 16);
            v19 = *(_QWORD *)(v16 + 24);
            if ( v14 <= *(_DWORD *)(v16 + 32) )
              break;
            v16 = *(_QWORD *)(v16 + 24);
            if ( !v19 )
              goto LABEL_13;
          }
          v17 = v16;
          v16 = *(_QWORD *)(v16 + 16);
        }
        while ( v18 );
LABEL_13:
        if ( v15 != v17 && v14 >= *(_DWORD *)(v17 + 32) )
          return *(_QWORD *)(v17 + 40);
      }
      v20 = (int *)&v48;
LABEL_16:
      v49[0] = v20;
      v17 = sub_39636C0(v13, v17, v49);
      return *(_QWORD *)(v17 + 40);
    }
  }
  v22 = a1[1];
  v23 = *(unsigned int *)(v22 + 24);
  if ( (_DWORD)v23 )
  {
    v24 = v23 - 1;
    v25 = *(_QWORD *)(v22 + 8);
    v26 = 1;
    v27 = (v23 - 1) & ((LODWORD(v46[0]) >> 9) ^ (LODWORD(v46[0]) >> 4));
    v28 = v27;
    v29 = (int *)(v25 + 16LL * v27);
    v30 = *(_QWORD *)v29;
    v31 = *(_QWORD *)v29;
    if ( v46[0] == *(_QWORD *)v29 )
    {
      if ( v29 != (int *)(v25 + 16 * v23) )
      {
LABEL_21:
        v32 = *((_QWORD *)v29 + 1);
LABEL_22:
        v48 = v32;
        v33 = sub_3967280(a1[2] + 264LL, &v48);
        v13 = (_QWORD *)*a1;
        v34 = *((_DWORD *)v33 + 2);
        v35 = *(_QWORD *)(*a1 + 16LL);
        v36 = *a1 + 8LL;
        v47 = v34;
        v17 = v36;
        if ( v35 )
        {
          do
          {
            while ( 1 )
            {
              v37 = *(_QWORD *)(v35 + 16);
              v38 = *(_QWORD *)(v35 + 24);
              if ( v34 <= *(_DWORD *)(v35 + 32) )
                break;
              v35 = *(_QWORD *)(v35 + 24);
              if ( !v38 )
                goto LABEL_27;
            }
            v17 = v35;
            v35 = *(_QWORD *)(v35 + 16);
          }
          while ( v37 );
LABEL_27:
          if ( v17 != v36 && v34 >= *(_DWORD *)(v17 + 32) )
            return *(_QWORD *)(v17 + 40);
        }
        v20 = &v47;
        goto LABEL_16;
      }
      return 0;
    }
    while ( 1 )
    {
      if ( v31 == -8 )
        return 0;
      v39 = v26 + 1;
      v28 = v24 & (v26 + v28);
      v40 = (__int64 *)(v25 + 16LL * v28);
      v31 = *v40;
      if ( v46[0] == *v40 )
        break;
      v26 = v39;
    }
    v41 = 1;
    v42 = 0;
    if ( v40 == (__int64 *)(v25 + 16LL * (unsigned int)v23) )
      return 0;
    while ( v30 != -8 )
    {
      if ( !v42 && v30 == -16 )
        v42 = v29;
      v27 = v24 & (v41 + v27);
      v29 = (int *)(v25 + 16LL * v27);
      v30 = *(_QWORD *)v29;
      if ( v46[0] == *(_QWORD *)v29 )
        goto LABEL_21;
      ++v41;
    }
    if ( !v42 )
      v42 = v29;
    v43 = *(_DWORD *)(v22 + 16);
    ++*(_QWORD *)v22;
    v44 = v43 + 1;
    if ( 4 * (v43 + 1) >= (unsigned int)(3 * v23) )
    {
      LODWORD(v23) = 2 * v23;
    }
    else if ( (int)v23 - *(_DWORD *)(v22 + 20) - v44 > (unsigned int)v23 >> 3 )
    {
LABEL_40:
      *(_DWORD *)(v22 + 16) = v44;
      if ( *(_QWORD *)v42 != -8 )
        --*(_DWORD *)(v22 + 20);
      v45 = v46[0];
      *((_QWORD *)v42 + 1) = 0;
      *(_QWORD *)v42 = v45;
      v32 = 0;
      goto LABEL_22;
    }
    sub_1447B20(v22, v23);
    sub_1AFF330(v22, v46, v49);
    v42 = v49[0];
    v44 = *(_DWORD *)(v22 + 16) + 1;
    goto LABEL_40;
  }
  return 0;
}
