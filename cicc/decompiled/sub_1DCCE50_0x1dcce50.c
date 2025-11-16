// Function: sub_1DCCE50
// Address: 0x1dcce50
//
__int64 __fastcall sub_1DCCE50(char *a1, int a2, __int64 a3)
{
  char *v4; // rax
  __int64 *v5; // r9
  char *v6; // r13
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // r15
  __int64 *v11; // rdi
  __int64 v12; // rbx
  __int64 v13; // rsi
  __int64 *v14; // r8
  __int64 *v15; // rax
  __int64 *v16; // rcx
  __int64 *v17; // r15
  __int64 *v18; // r12
  __int64 *v19; // r9
  __int64 v20; // rbx
  __int64 *v21; // rax
  unsigned int v22; // ecx
  unsigned int v23; // edi
  unsigned int v24; // esi
  __int64 *v25; // rax
  __int64 *v26; // r14
  unsigned int v27; // r12d
  __int64 *v29; // rdx
  __int64 *v30; // rax
  __int64 *v31; // [rsp+8h] [rbp-A8h]
  __int64 v32; // [rsp+10h] [rbp-A0h] BYREF
  __int64 *v33; // [rsp+18h] [rbp-98h]
  __int64 *v34; // [rsp+20h] [rbp-90h]
  __int64 v35; // [rsp+28h] [rbp-88h]
  int v36; // [rsp+30h] [rbp-80h]
  _BYTE v37[120]; // [rsp+38h] [rbp-78h] BYREF

  v4 = sub_1DCC790(a1, a2);
  v36 = 0;
  v5 = (__int64 *)v37;
  v6 = v4;
  v7 = *((_QWORD *)v4 + 4);
  v32 = 0;
  v8 = *((_QWORD *)v6 + 5);
  v33 = (__int64 *)v37;
  v34 = (__int64 *)v37;
  v35 = 8;
  v9 = (v8 - v7) >> 3;
  if ( (_DWORD)v9 )
  {
    v10 = 0;
    v11 = (__int64 *)v37;
    v12 = 8LL * (unsigned int)(v9 - 1);
    while ( 1 )
    {
      v13 = *(_QWORD *)(*(_QWORD *)(v7 + v10) + 24LL);
      if ( v11 != v5 )
        break;
      v14 = &v11[HIDWORD(v35)];
      if ( v14 == v11 )
      {
LABEL_60:
        if ( HIDWORD(v35) >= (unsigned int)v35 )
          break;
        ++HIDWORD(v35);
        *v14 = v13;
        v5 = v33;
        ++v32;
        v11 = v34;
      }
      else
      {
        v15 = v11;
        v16 = 0;
        while ( v13 != *v15 )
        {
          if ( *v15 == -2 )
            v16 = v15;
          if ( v14 == ++v15 )
          {
            if ( !v16 )
              goto LABEL_60;
            *v16 = v13;
            v11 = v34;
            --v36;
            v5 = v33;
            ++v32;
            if ( v12 != v10 )
              goto LABEL_5;
            goto LABEL_15;
          }
        }
      }
LABEL_4:
      if ( v12 == v10 )
      {
LABEL_15:
        v17 = *(__int64 **)(a3 + 88);
        v31 = *(__int64 **)(a3 + 96);
        if ( v31 == v17 )
        {
          v27 = 0;
          goto LABEL_32;
        }
        goto LABEL_16;
      }
LABEL_5:
      v7 = *((_QWORD *)v6 + 4);
      v10 += 8;
    }
    sub_16CCBA0((__int64)&v32, v13);
    v11 = v34;
    v5 = v33;
    goto LABEL_4;
  }
  v30 = *(__int64 **)(a3 + 96);
  v17 = *(__int64 **)(a3 + 88);
  v27 = 0;
  v31 = v30;
  if ( v17 == v30 )
    return v27;
LABEL_16:
  v18 = (__int64 *)(v6 + 8);
  while ( 1 )
  {
    v19 = (__int64 *)*((_QWORD *)v6 + 1);
    v20 = *v17;
    if ( v19 == v18 )
      goto LABEL_26;
    v21 = *(__int64 **)v6;
    v22 = *(_DWORD *)(v20 + 48);
    if ( *(__int64 **)v6 == v18 )
      break;
    v23 = *((_DWORD *)v21 + 4);
    v24 = v22 >> 7;
    if ( v22 >> 7 == v23 )
      goto LABEL_50;
LABEL_20:
    if ( v24 < v23 )
    {
      if ( v21 == v19 )
      {
        *(_QWORD *)v6 = v21;
      }
      else
      {
        do
          v21 = (__int64 *)v21[1];
        while ( v19 != v21 && v24 < *((_DWORD *)v21 + 4) );
LABEL_24:
        *(_QWORD *)v6 = v21;
        if ( v21 == v18 )
          goto LABEL_26;
      }
      if ( v24 != *((_DWORD *)v21 + 4) )
        goto LABEL_26;
      goto LABEL_50;
    }
    if ( v21 != v18 )
    {
      while ( v24 > v23 )
      {
        v21 = (__int64 *)*v21;
        if ( v21 == v18 )
          goto LABEL_59;
        v23 = *((_DWORD *)v21 + 4);
      }
      goto LABEL_24;
    }
LABEL_59:
    *(_QWORD *)v6 = v18;
LABEL_26:
    v11 = v34;
    v25 = v33;
    if ( v34 == v33 )
    {
      v26 = &v34[HIDWORD(v35)];
      if ( v34 == v26 )
      {
        v29 = v34;
      }
      else
      {
        do
        {
          if ( v20 == *v25 )
            break;
          ++v25;
        }
        while ( v26 != v25 );
        v29 = &v34[HIDWORD(v35)];
      }
      goto LABEL_44;
    }
    v26 = &v34[(unsigned int)v35];
    v25 = sub_16CC9F0((__int64)&v32, v20);
    if ( v20 == *v25 )
    {
      v11 = v34;
      if ( v34 == v33 )
        v29 = &v34[HIDWORD(v35)];
      else
        v29 = &v34[(unsigned int)v35];
LABEL_44:
      while ( v29 != v25 && (unsigned __int64)*v25 >= 0xFFFFFFFFFFFFFFFELL )
        ++v25;
      goto LABEL_30;
    }
    v11 = v34;
    if ( v34 == v33 )
    {
      v25 = &v34[HIDWORD(v35)];
      v29 = v25;
      goto LABEL_44;
    }
    v25 = &v34[(unsigned int)v35];
LABEL_30:
    if ( v26 != v25 )
    {
      v5 = v33;
      v27 = 1;
      goto LABEL_32;
    }
    if ( v31 == ++v17 )
    {
      v5 = v33;
      v27 = 0;
      goto LABEL_32;
    }
  }
  v21 = (__int64 *)*((_QWORD *)v6 + 2);
  v24 = v22 >> 7;
  *(_QWORD *)v6 = v21;
  v23 = *((_DWORD *)v21 + 4);
  if ( v22 >> 7 != v23 )
    goto LABEL_20;
  if ( v21 == v18 )
    goto LABEL_26;
LABEL_50:
  if ( (v21[((v22 >> 6) & 1) + 3] & (1LL << v22)) == 0 )
    goto LABEL_26;
  v11 = v34;
  v5 = v33;
  v27 = 1;
LABEL_32:
  if ( v5 != v11 )
    _libc_free((unsigned __int64)v11);
  return v27;
}
