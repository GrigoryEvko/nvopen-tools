// Function: sub_2BBD7D0
// Address: 0x2bbd7d0
//
__int64 __fastcall sub_2BBD7D0(__int64 a1, __int64 a2, _DWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // eax
  char v7; // al
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r15
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r9
  __int64 v15; // rdx
  int v16; // eax
  __int64 v17; // rcx
  __int64 v18; // rbx
  __int64 v19; // rdx
  __int64 v20; // rax
  unsigned int v21; // esi
  int v22; // eax
  __int64 v23; // rdx
  __int64 v24; // r8
  __int64 v25; // rcx
  bool v26; // zf
  __int64 v27; // rdx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // r15
  __int64 v31; // r14
  __int64 v32; // r14
  unsigned __int64 v33; // r15
  __int64 v34; // r12
  unsigned __int64 *v35; // r12
  int v36; // r12d
  int v37; // eax
  char v40; // [rsp+18h] [rbp-C8h]
  __int64 v41; // [rsp+18h] [rbp-C8h]
  __int64 v42; // [rsp+18h] [rbp-C8h]
  __int64 v43; // [rsp+18h] [rbp-C8h]
  __int64 v44; // [rsp+18h] [rbp-C8h]
  __int64 v45; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v46[3]; // [rsp+28h] [rbp-B8h] BYREF
  int v47; // [rsp+40h] [rbp-A0h]
  char *v48; // [rsp+50h] [rbp-90h] BYREF
  __int64 v49; // [rsp+58h] [rbp-88h]
  _BYTE v50[16]; // [rsp+60h] [rbp-80h] BYREF
  int v51; // [rsp+70h] [rbp-70h]
  char *v52; // [rsp+80h] [rbp-60h] BYREF
  __int64 v53; // [rsp+88h] [rbp-58h]
  _BYTE v54[16]; // [rsp+90h] [rbp-50h] BYREF
  int v55; // [rsp+A0h] [rbp-40h]

  v6 = *(_DWORD *)(a2 + 8);
  v48 = v50;
  v49 = 0x400000000LL;
  if ( v6 )
  {
    sub_2B0D430((__int64)&v48, a2, (__int64)a3, a4, (__int64)&v48, a6);
    v6 = v49;
    v51 = 0;
    v52 = v54;
    v53 = 0x400000000LL;
    if ( (_DWORD)v49 )
    {
      sub_2B0D510((__int64)&v52, &v48, v12, v13, (__int64)&v48, v14);
      v6 = v51;
    }
  }
  else
  {
    v51 = 0;
    v52 = v54;
    v53 = 0x400000000LL;
  }
  v55 = v6;
  v7 = sub_2B4D8D0(a1, (const void **)&v52, &v45);
  v10 = v45;
  if ( v7 )
  {
    v40 = 0;
    goto LABEL_5;
  }
  v21 = *(_DWORD *)(a1 + 24);
  v22 = *(_DWORD *)(a1 + 16);
  v46[0] = v45;
  ++*(_QWORD *)a1;
  v23 = (unsigned int)(v22 + 1);
  v24 = 2 * v21;
  if ( 4 * (int)v23 >= 3 * v21 )
  {
    v21 *= 2;
  }
  else
  {
    v25 = v21 - *(_DWORD *)(a1 + 20) - (unsigned int)v23;
    if ( (unsigned int)v25 > v21 >> 3 )
      goto LABEL_22;
  }
  sub_2BBD510(a1, v21);
  sub_2B4D8D0(a1, (const void **)&v52, v46);
  v10 = v46[0];
  v23 = (unsigned int)(*(_DWORD *)(a1 + 16) + 1);
LABEL_22:
  *(_DWORD *)(a1 + 16) = v23;
  v26 = *(_DWORD *)(v10 + 8) == 1;
  v47 = -2;
  if ( !v26 || (v23 = *(_QWORD *)v10, **(_DWORD **)v10 != -2) )
    --*(_DWORD *)(a1 + 20);
  sub_2B0D510(v10, &v52, v23, v25, v24, v9);
  v40 = 1;
  *(_DWORD *)(v10 + 32) = v55;
LABEL_5:
  if ( v52 != v54 )
    _libc_free((unsigned __int64)v52);
  if ( v48 != v50 )
    _libc_free((unsigned __int64)v48);
  if ( !v40 )
    return *(_QWORD *)(a1 + 32) + 40LL * *(unsigned int *)(v10 + 32);
  *(_DWORD *)(v10 + 32) = *(_DWORD *)(a1 + 40);
  v15 = *(unsigned int *)(a1 + 40);
  v16 = v15;
  if ( *(_DWORD *)(a1 + 44) <= (unsigned int)v15 )
  {
    v17 = sub_C8D7D0(a1 + 32, a1 + 48, 0, 0x28u, (unsigned __int64 *)&v52, v9);
    v30 = 40LL * *(unsigned int *)(a1 + 40);
    v31 = v30 + v17;
    if ( v30 + v17 )
    {
      *(_QWORD *)v31 = v31 + 16;
      *(_QWORD *)(v31 + 8) = 0x400000000LL;
      if ( *(_DWORD *)(a2 + 8) )
      {
        v44 = v17;
        sub_2B0D510(v31, (char **)a2, v27, v17, v28, v29);
        v17 = v44;
      }
      *(_DWORD *)(v31 + 32) = *a3;
      v30 = 40LL * *(unsigned int *)(a1 + 40);
    }
    v32 = *(_QWORD *)(a1 + 32);
    v33 = v32 + v30;
    if ( v32 != v33 )
    {
      v34 = v17;
      do
      {
        if ( v34 )
        {
          *(_DWORD *)(v34 + 8) = 0;
          *(_QWORD *)v34 = v34 + 16;
          *(_DWORD *)(v34 + 12) = 4;
          if ( *(_DWORD *)(v32 + 8) )
          {
            v41 = v17;
            sub_2B0D510(v34, (char **)v32, v27, v17, v28, v29);
            v17 = v41;
          }
          *(_DWORD *)(v34 + 32) = *(_DWORD *)(v32 + 32);
        }
        v32 += 40;
        v34 += 40;
      }
      while ( v33 != v32 );
      v33 = *(_QWORD *)(a1 + 32);
      v35 = (unsigned __int64 *)(v33 + 40LL * *(unsigned int *)(a1 + 40));
      if ( v35 != (unsigned __int64 *)v33 )
      {
        do
        {
          v35 -= 5;
          if ( (unsigned __int64 *)*v35 != v35 + 2 )
          {
            v42 = v17;
            _libc_free(*v35);
            v17 = v42;
          }
        }
        while ( (unsigned __int64 *)v33 != v35 );
        v33 = *(_QWORD *)(a1 + 32);
      }
    }
    v36 = (int)v52;
    if ( v33 != a1 + 48 )
    {
      v43 = v17;
      _libc_free(v33);
      v17 = v43;
    }
    v37 = *(_DWORD *)(a1 + 40);
    *(_QWORD *)(a1 + 32) = v17;
    *(_DWORD *)(a1 + 44) = v36;
    v20 = (unsigned int)(v37 + 1);
    *(_DWORD *)(a1 + 40) = v20;
  }
  else
  {
    v17 = *(_QWORD *)(a1 + 32);
    v18 = v17 + 40 * v15;
    if ( v18 )
    {
      *(_QWORD *)v18 = v18 + 16;
      *(_QWORD *)(v18 + 8) = 0x400000000LL;
      v19 = *(unsigned int *)(a2 + 8);
      if ( (_DWORD)v19 )
        sub_2B0D510(v18, (char **)a2, v19, v17, v8, v9);
      *(_DWORD *)(v18 + 32) = *a3;
      v16 = *(_DWORD *)(a1 + 40);
      v17 = *(_QWORD *)(a1 + 32);
    }
    v20 = (unsigned int)(v16 + 1);
    *(_DWORD *)(a1 + 40) = v20;
  }
  return v17 + 40 * v20 - 40;
}
