// Function: sub_18509A0
// Address: 0x18509a0
//
__int64 __fastcall sub_18509A0(__int64 a1, __int64 a2)
{
  __int64 **v2; // rax
  __int64 **v3; // r15
  __int64 **v4; // r12
  char v5; // bl
  __int64 v6; // rdx
  __int64 v7; // rcx
  _QWORD *v8; // r8
  _QWORD *v9; // r9
  __int64 v10; // rdi
  char v11; // al
  __int64 *v12; // r12
  __int64 v13; // r15
  __int64 v14; // r13
  int v15; // eax
  __int64 *v16; // r12
  __int64 *v17; // r13
  char v18; // dl
  __int64 v19; // r15
  __int64 *v20; // rdi
  unsigned int v21; // r12d
  unsigned __int8 v23; // al
  int v24; // eax
  int v25; // eax
  int v26; // ebx
  int v27; // ebx
  __int64 *v28; // [rsp+0h] [rbp-5A0h]
  char v29; // [rsp+Dh] [rbp-593h]
  char v30; // [rsp+Eh] [rbp-592h]
  unsigned __int8 v31; // [rsp+Fh] [rbp-591h]
  __int64 v32; // [rsp+10h] [rbp-590h]
  __int64 v33; // [rsp+28h] [rbp-578h] BYREF
  __int64 v34; // [rsp+30h] [rbp-570h] BYREF
  __int64 v35; // [rsp+38h] [rbp-568h]
  __int64 v36; // [rsp+40h] [rbp-560h] BYREF
  __int64 *v37; // [rsp+80h] [rbp-520h] BYREF
  __int64 v38; // [rsp+88h] [rbp-518h]
  _BYTE v39[64]; // [rsp+90h] [rbp-510h] BYREF
  __int64 v40[10]; // [rsp+D0h] [rbp-4D0h] BYREF
  char v41; // [rsp+120h] [rbp-480h]
  __int64 v42; // [rsp+128h] [rbp-478h]
  __int64 v43; // [rsp+3F0h] [rbp-1B0h]
  unsigned __int64 v44; // [rsp+3F8h] [rbp-1A8h]
  __int64 v45; // [rsp+458h] [rbp-148h]
  unsigned __int64 v46; // [rsp+460h] [rbp-140h]
  char v47; // [rsp+4F8h] [rbp-A8h]
  __int64 v48[12]; // [rsp+500h] [rbp-A0h] BYREF
  char v49; // [rsp+560h] [rbp-40h]

  v2 = (__int64 **)&v36;
  v40[0] = a1;
  v47 = 0;
  v49 = 0;
  v34 = 0;
  v35 = 1;
  do
    *v2++ = (__int64 *)-8LL;
  while ( v2 != &v37 );
  v3 = *(__int64 ***)(a2 + 16);
  v4 = *(__int64 ***)(a2 + 24);
  v37 = (__int64 *)v39;
  v38 = 0x800000000LL;
  if ( v3 == v4 )
  {
    v21 = 0;
    goto LABEL_30;
  }
  v5 = 0;
  do
  {
    while ( 1 )
    {
      v10 = **v3;
      v33 = v10;
      if ( !v10 )
        break;
      a2 = 35;
      if ( (unsigned __int8)sub_1560180(v10 + 112, 35) )
        break;
      a2 = 18;
      if ( (unsigned __int8)sub_1560180(v33 + 112, 18) )
        break;
      a2 = (__int64)&v33;
      ++v3;
      sub_18506C0((__int64)&v34, &v33, v6, v7, v8, v9);
      if ( v4 == v3 )
        goto LABEL_10;
    }
    ++v3;
    v5 = 1;
  }
  while ( v4 != v3 );
LABEL_10:
  if ( !(_DWORD)v38 )
  {
    v20 = v37;
    v21 = 0;
    goto LABEL_28;
  }
  v11 = sub_18487A0((__int64)&v34, a2);
  v12 = v37;
  v30 = 0;
  v29 = v11;
  v28 = &v37[(unsigned int)v38];
  if ( v37 == v28 )
    goto LABEL_48;
  do
  {
    v13 = *v12;
    v14 = sub_1833A50((__int64)v40, *v12);
    if ( sub_15E4F60(v13) )
    {
      v15 = sub_134CE70(v14, v13);
      if ( v15 == 4 )
        goto LABEL_16;
LABEL_14:
      if ( (v15 & 2) != 0 )
        goto LABEL_48;
LABEL_15:
      v30 = 1;
      goto LABEL_16;
    }
    sub_15E4B50(v13);
    v31 = v23;
    if ( v23 )
    {
      v15 = sub_134CE70(v14, v13);
      if ( v15 != 4 )
        goto LABEL_14;
    }
    else if ( (unsigned int)sub_134CE70(v14, v13) != 4 )
    {
      v24 = sub_1849A60(v13, v14, (__int64)&v34);
      if ( v24 == 1 )
        goto LABEL_15;
      if ( v24 == 2 )
        goto LABEL_40;
    }
LABEL_16:
    ++v12;
  }
  while ( v28 != v12 );
  v16 = v37;
  v17 = &v37[(unsigned int)v38];
  if ( v37 != v17 )
  {
    v31 = 0;
    while ( 1 )
    {
      v19 = *v16;
      v32 = *v16 + 112;
      if ( (unsigned __int8)sub_1560180(v32, 36) )
        goto LABEL_22;
      if ( !(unsigned __int8)sub_1560180(v32, 36) && !(unsigned __int8)sub_1560180(v32, 37) )
        break;
      if ( !v30 )
      {
        sub_15E0E50(v19, -1, 37);
        sub_15E0E50(v19, -1, 36);
        v18 = 36;
LABEL_21:
        sub_15E0D50(v19, -1, v18);
        v31 = 1;
      }
LABEL_22:
      if ( v17 == ++v16 )
        goto LABEL_40;
    }
    sub_15E0E50(v19, -1, 37);
    sub_15E0E50(v19, -1, 36);
    v18 = 37 - (v30 == 0);
    goto LABEL_21;
  }
LABEL_48:
  v31 = 0;
LABEL_40:
  v25 = sub_184E230((__int64)&v34);
  v21 = v25 | v31;
  LOBYTE(v21) = v29 | v25 | v31;
  if ( !v5 )
  {
    v26 = sub_184FD30((__int64)&v34);
    v27 = sub_184F780((__int64)&v34) | v26;
    v21 |= sub_184B020((__int64)&v34) | v27;
    if ( (_DWORD)v38 == 1 )
      v21 |= sub_18494B0((__int64)&v34);
  }
  v20 = v37;
LABEL_28:
  if ( v20 != (__int64 *)v39 )
    _libc_free((unsigned __int64)v20);
LABEL_30:
  if ( (v35 & 1) == 0 )
    j___libc_free_0(v36);
  if ( v49 )
    sub_134CA00(v48);
  if ( v47 )
  {
    if ( v46 != v45 )
      _libc_free(v46);
    if ( v44 != v43 )
      _libc_free(v44);
    if ( (v41 & 1) == 0 )
      j___libc_free_0(v42);
  }
  return v21;
}
