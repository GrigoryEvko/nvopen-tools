// Function: sub_137A080
// Address: 0x137a080
//
__int64 __fastcall sub_137A080(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r12
  unsigned int i; // r14d
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  int v9; // r14d
  int v10; // r12d
  int *v11; // r15
  int v12; // edx
  int *v13; // rdi
  int *v14; // r14
  int *v15; // r15
  int v16; // edx
  unsigned int v17; // r12d
  int *v19; // r14
  int *v20; // r12
  int v21; // edx
  int v22; // [rsp+18h] [rbp-88h]
  int *v23; // [rsp+18h] [rbp-88h]
  int v24; // [rsp+2Ch] [rbp-74h] BYREF
  int *v25; // [rsp+30h] [rbp-70h] BYREF
  __int64 v26; // [rsp+38h] [rbp-68h]
  _BYTE v27[16]; // [rsp+40h] [rbp-60h] BYREF
  int *v28; // [rsp+50h] [rbp-50h] BYREF
  __int64 v29; // [rsp+58h] [rbp-48h]
  _BYTE v30[64]; // [rsp+60h] [rbp-40h] BYREF

  v25 = (int *)v27;
  v26 = 0x400000000LL;
  v28 = (int *)v30;
  v29 = 0x400000000LL;
  v3 = sub_157EBA0(a2);
  if ( !v3 )
    goto LABEL_26;
  v4 = v3;
  v22 = sub_15F4D60(v3);
  if ( v22 )
  {
    for ( i = 0; i != v22; ++i )
    {
      while ( 1 )
      {
        v7 = sub_15F4DF0(v4, i);
        if ( sub_1377F70(a1 + 240, v7) )
          break;
        v8 = (unsigned int)v29;
        if ( (unsigned int)v29 >= HIDWORD(v29) )
        {
          sub_16CD150(&v28, v30, 0, 4);
          v8 = (unsigned int)v29;
        }
        v28[v8] = i++;
        LODWORD(v29) = v29 + 1;
        if ( i == v22 )
          goto LABEL_11;
      }
      v6 = (unsigned int)v26;
      if ( (unsigned int)v26 >= HIDWORD(v26) )
      {
        sub_16CD150(&v25, v27, 0, 4);
        v6 = (unsigned int)v26;
      }
      v25[v6] = i;
      LODWORD(v26) = v26 + 1;
    }
  }
LABEL_11:
  if ( (_DWORD)v26 )
  {
    if ( !(_DWORD)v29 )
    {
      sub_16AF710(&v24, 1, (unsigned int)v26);
      v19 = v25;
      v20 = &v25[(unsigned int)v26];
      if ( v25 != v20 )
      {
        do
        {
          v21 = *v19++;
          sub_1379150(a1, a2, v21, v24);
        }
        while ( v20 != v19 );
      }
      goto LABEL_17;
    }
    v9 = sub_16AF730(4, 68LL * (unsigned int)v26);
    v10 = sub_16AF730(64, 68LL * (unsigned int)v29);
    v11 = v25;
    v23 = &v25[(unsigned int)v26];
    if ( v23 != v25 )
    {
      do
      {
        v12 = *v11++;
        sub_1379150(a1, a2, v12, v9);
      }
      while ( v23 != v11 );
    }
    v13 = v28;
    v14 = &v28[(unsigned int)v29];
    v15 = v28;
    if ( v14 != v28 )
    {
      do
      {
        v16 = *v15++;
        sub_1379150(a1, a2, v16, v10);
      }
      while ( v14 != v15 );
LABEL_17:
      v13 = v28;
      v17 = 1;
      goto LABEL_18;
    }
    v17 = 1;
  }
  else
  {
LABEL_26:
    v13 = v28;
    v17 = 0;
  }
LABEL_18:
  if ( v13 != (int *)v30 )
    _libc_free((unsigned __int64)v13);
  if ( v25 != (int *)v27 )
    _libc_free((unsigned __int64)v25);
  return v17;
}
