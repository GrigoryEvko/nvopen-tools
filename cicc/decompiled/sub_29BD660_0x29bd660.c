// Function: sub_29BD660
// Address: 0x29bd660
//
__int64 __fastcall sub_29BD660(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v8; // al
  char v9; // al
  __int64 v10; // r15
  __int64 v11; // r15
  unsigned int v12; // esi
  __int64 v13; // rax
  unsigned int v14; // ecx
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  unsigned int v20; // r12d
  __int64 v22; // rax
  __int64 *v23; // rbx
  __int64 *v24; // r13
  __int64 *v25; // r15
  __int64 *v26; // r14
  __int64 *v27; // r15
  __int64 *v28; // r15
  __int64 *v29; // r15
  signed __int64 v30; // rax
  __int64 *v31; // r13
  __int64 *v32; // r13
  __int64 *v33; // r13
  __int64 *v34; // [rsp+8h] [rbp-D8h]
  __int64 *v35; // [rsp+10h] [rbp-D0h] BYREF
  unsigned int v36; // [rsp+18h] [rbp-C8h]
  char v37; // [rsp+20h] [rbp-C0h] BYREF
  unsigned __int8 v38; // [rsp+50h] [rbp-90h]
  __int64 *v39; // [rsp+60h] [rbp-80h] BYREF
  unsigned int v40; // [rsp+68h] [rbp-78h]
  char v41; // [rsp+70h] [rbp-70h] BYREF
  unsigned __int8 v42; // [rsp+A0h] [rbp-40h]

  if ( a1 == a2 )
    return 1;
  if ( (unsigned __int8)sub_B19720(a3, a1, a2) )
  {
    sub_B19AA0(a4, a2, a1);
    if ( v8 )
      return 1;
  }
  sub_B19AA0(a4, a1, a2);
  if ( v9 )
  {
    if ( (unsigned __int8)sub_B19720(a3, a2, a1) )
      return 1;
  }
  v10 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 80LL);
  if ( !v10 || (v11 = v10 - 24, a1 != v11) && a2 != v11 )
  {
    v12 = *(_DWORD *)(a3 + 32);
    v13 = (unsigned int)(*(_DWORD *)(a1 + 44) + 1);
    v14 = *(_DWORD *)(a2 + 44) + 1;
    if ( (unsigned int)v13 >= v12 )
    {
      v18 = v14;
      if ( v12 <= v14 )
        BUG();
      v15 = *(_QWORD *)(a3 + 24);
      v17 = 0;
    }
    else
    {
      v15 = *(_QWORD *)(a3 + 24);
      v16 = 0;
      v17 = *(_QWORD *)(v15 + 8 * v13);
      if ( v12 <= v14 )
        goto LABEL_15;
      v18 = v14;
    }
    v16 = *(_QWORD *)(v15 + 8 * v18);
    while ( v17 != v16 )
    {
      if ( *(_DWORD *)(v17 + 16) < *(_DWORD *)(v16 + 16) )
      {
        v19 = v17;
        v17 = v16;
        v16 = v19;
      }
      v17 = *(_QWORD *)(v17 + 8);
LABEL_15:
      ;
    }
    v11 = *(_QWORD *)v16;
  }
  sub_29BD470((__int64)&v35, a1, v11, a3, a4);
  v20 = v38;
  if ( v38 )
  {
    sub_29BD470((__int64)&v39, a2, v11, a3, a4);
    v20 = v42;
    if ( !v42 )
    {
LABEL_19:
      if ( v38 && v35 != (__int64 *)&v37 )
        _libc_free((unsigned __int64)v35);
      return v20;
    }
    v22 = v36;
    if ( !v36 )
    {
      LOBYTE(v20) = v40 == 0;
      goto LABEL_27;
    }
    if ( v36 != v40 )
    {
      v20 = 0;
      goto LABEL_27;
    }
    v23 = v35;
    v34 = &v35[v36];
    if ( (8LL * v36) >> 5 )
    {
      v24 = &v35[4 * ((8LL * v36) >> 5)];
      while ( 1 )
      {
        v29 = &v39[v22];
        if ( v29 == sub_29BD070(v39, v29, v23) )
          goto LABEL_40;
        v25 = &v39[v40];
        if ( v25 == sub_29BD070(v39, v25, v23 + 1) )
        {
          LOBYTE(v20) = v23 + 1 == v34;
          goto LABEL_41;
        }
        v26 = v23 + 2;
        v27 = &v39[v40];
        if ( v27 == sub_29BD070(v39, v27, v23 + 2)
          || (v26 = v23 + 3, v28 = &v39[v40], v28 == sub_29BD070(v39, v28, v23 + 3)) )
        {
          LOBYTE(v20) = v34 == v26;
          goto LABEL_41;
        }
        v23 += 4;
        if ( v24 == v23 )
          break;
        v22 = v40;
      }
    }
    v30 = (char *)v34 - (char *)v23;
    if ( (char *)v34 - (char *)v23 != 16 )
    {
      if ( v30 != 24 )
      {
        if ( v30 != 8 )
          goto LABEL_41;
        goto LABEL_48;
      }
      v32 = &v39[v40];
      if ( v32 == sub_29BD070(v39, v32, v23) )
      {
LABEL_40:
        LOBYTE(v20) = v34 == v23;
        goto LABEL_41;
      }
      ++v23;
    }
    v33 = &v39[v40];
    if ( v33 != sub_29BD070(v39, v33, v23) )
    {
      ++v23;
LABEL_48:
      v31 = &v39[v40];
      if ( v31 == sub_29BD070(v39, v31, v23) )
        goto LABEL_40;
LABEL_41:
      if ( !v42 )
        goto LABEL_19;
LABEL_27:
      if ( v39 != (__int64 *)&v41 )
        _libc_free((unsigned __int64)v39);
      goto LABEL_19;
    }
    goto LABEL_40;
  }
  return v20;
}
