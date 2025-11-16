// Function: sub_38990C0
// Address: 0x38990c0
//
__int64 *__fastcall sub_38990C0(_QWORD *a1, unsigned int a2, __int64 a3, unsigned __int64 a4)
{
  __int64 v8; // rdx
  __int64 v9; // rax
  _QWORD *v10; // r15
  _QWORD *v11; // rsi
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // r11
  __int64 v15; // r10
  __int64 v16; // rax
  __int64 *v17; // r13
  __int64 v18; // rax
  _QWORD *v19; // r8
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  char v24; // di
  unsigned __int64 v25; // rdi
  __int64 *v26; // rax
  __int64 v27; // [rsp+0h] [rbp-130h]
  __int64 v28; // [rsp+8h] [rbp-128h]
  __int64 v29; // [rsp+8h] [rbp-128h]
  __int64 v30; // [rsp+10h] [rbp-120h]
  __int64 v31; // [rsp+10h] [rbp-120h]
  __int64 v32; // [rsp+10h] [rbp-120h]
  unsigned int v33; // [rsp+10h] [rbp-120h]
  __int64 v34; // [rsp+18h] [rbp-118h]
  __int64 v35; // [rsp+18h] [rbp-118h]
  __int64 v36; // [rsp+20h] [rbp-110h]
  _QWORD v37[2]; // [rsp+40h] [rbp-F0h] BYREF
  __int16 v38; // [rsp+50h] [rbp-E0h]
  _QWORD v39[2]; // [rsp+60h] [rbp-D0h] BYREF
  __int16 v40; // [rsp+70h] [rbp-C0h]
  _QWORD v41[2]; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v42; // [rsp+90h] [rbp-A0h]
  char *v43; // [rsp+A0h] [rbp-90h]
  char v44; // [rsp+B0h] [rbp-80h]
  char v45; // [rsp+B1h] [rbp-7Fh]
  _QWORD v46[2]; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v47; // [rsp+D0h] [rbp-60h]
  __int64 v48[2]; // [rsp+E0h] [rbp-50h] BYREF
  _WORD v49[32]; // [rsp+F0h] [rbp-40h] BYREF

  if ( *(_BYTE *)(a3 + 8) != 15 )
  {
    v49[0] = 259;
    v17 = 0;
    v48[0] = (__int64)"global variable reference must have pointer type";
    sub_38814C0((__int64)(a1 + 1), a4, (__int64)v48);
    return v17;
  }
  v8 = a1[125];
  if ( a2 < (unsigned __int64)((a1[126] - v8) >> 3) )
  {
    v17 = *(__int64 **)(v8 + 8LL * a2);
    if ( v17 )
    {
      if ( a3 == *v17 )
        return v17;
      goto LABEL_27;
    }
  }
  v9 = a1[121];
  v10 = a1 + 120;
  if ( v9 )
  {
    v11 = a1 + 120;
    do
    {
      while ( 1 )
      {
        v12 = *(_QWORD *)(v9 + 16);
        v13 = *(_QWORD *)(v9 + 24);
        if ( *(_DWORD *)(v9 + 32) >= a2 )
          break;
        v9 = *(_QWORD *)(v9 + 24);
        if ( !v13 )
          goto LABEL_8;
      }
      v11 = (_QWORD *)v9;
      v9 = *(_QWORD *)(v9 + 16);
    }
    while ( v12 );
LABEL_8:
    if ( v10 != v11 && *((_DWORD *)v11 + 8) <= a2 )
    {
      v17 = (__int64 *)v11[5];
      if ( v17 )
      {
        if ( a3 == *v17 )
          return v17;
LABEL_27:
        v45 = 1;
        v43 = "'";
        v44 = 3;
        sub_3888960(v48, *v17);
        LODWORD(v36) = a2;
        v37[0] = "'@";
        v38 = 2307;
        v37[1] = v36;
        v39[0] = v37;
        v39[1] = "' defined with type '";
        v41[0] = v39;
        v40 = 770;
        v41[1] = v48;
        LOWORD(v42) = 1026;
        v46[1] = "'";
        v46[0] = v41;
        LOWORD(v47) = 770;
        sub_38814C0((__int64)(a1 + 1), a4, (__int64)v46);
        if ( (_WORD *)v48[0] != v49 )
          j_j___libc_free_0(v48[0]);
        return 0;
      }
    }
  }
  v30 = a3;
  v48[0] = (__int64)v49;
  sub_3887410(v48, byte_3F871B3, (__int64)byte_3F871B3);
  v14 = a1[22];
  v15 = *(_QWORD *)(v30 + 24);
  if ( *(_BYTE *)(v15 + 8) == 12 )
  {
    v46[0] = v48;
    v28 = v15;
    v31 = v14;
    LOWORD(v47) = 260;
    v16 = sub_1648B60(120);
    v17 = (__int64 *)v16;
    if ( v16 )
      sub_15E2490(v16, v28, 9, (__int64)v46, v31);
  }
  else
  {
    v46[0] = v48;
    LOWORD(v47) = 260;
    v27 = v15;
    v29 = v14;
    v33 = *(_DWORD *)(v30 + 8) >> 8;
    v26 = sub_1648A60(88, 1u);
    v17 = v26;
    if ( v26 )
      sub_15E51E0((__int64)v26, v29, v27, 0, 9, 0, (__int64)v46, 0, 0, v33, 0);
  }
  if ( (_WORD *)v48[0] != v49 )
    j_j___libc_free_0(v48[0]);
  v18 = a1[121];
  v19 = a1 + 120;
  if ( !v18 )
    goto LABEL_32;
  do
  {
    if ( *(_DWORD *)(v18 + 32) < a2 )
    {
      v18 = *(_QWORD *)(v18 + 24);
    }
    else
    {
      v19 = (_QWORD *)v18;
      v18 = *(_QWORD *)(v18 + 16);
    }
  }
  while ( v18 );
  if ( v10 == v19 || *((_DWORD *)v19 + 8) > a2 )
  {
LABEL_32:
    v32 = (__int64)v19;
    v21 = sub_22077B0(0x38u);
    *(_DWORD *)(v21 + 32) = a2;
    *(_QWORD *)(v21 + 40) = 0;
    *(_QWORD *)(v21 + 48) = 0;
    v34 = v21;
    v22 = sub_3898FC0(a1 + 119, v32, (unsigned int *)(v21 + 32));
    if ( v23 )
    {
      v24 = v10 == (_QWORD *)v23 || v22 || a2 < *(_DWORD *)(v23 + 32);
      sub_220F040(v24, v34, (_QWORD *)v23, a1 + 120);
      v19 = (_QWORD *)v34;
      ++a1[124];
    }
    else
    {
      v25 = v34;
      v35 = v22;
      j_j___libc_free_0(v25);
      v19 = (_QWORD *)v35;
    }
  }
  v19[5] = v17;
  v19[6] = a4;
  return v17;
}
