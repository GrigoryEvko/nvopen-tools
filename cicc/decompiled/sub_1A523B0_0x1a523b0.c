// Function: sub_1A523B0
// Address: 0x1a523b0
//
__int64 __fastcall sub_1A523B0(__int64 a1, __int64 a2)
{
  __int64 *v2; // r12
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 *v5; // r15
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // r15
  unsigned int v8; // r12d
  __int64 v9; // rax
  unsigned __int64 v10; // rsi
  __int64 v11; // rbx
  __int64 *v12; // rax
  __int64 *v13; // r13
  int v14; // ecx
  int v15; // ecx
  __int64 v16; // r9
  unsigned int v17; // edi
  __int64 *v18; // rax
  __int64 v19; // r11
  __int64 **v20; // rax
  unsigned int v21; // r13d
  int v22; // eax
  __int64 *v24; // rdi
  __int64 *v25; // rax
  __int64 *v26; // rcx
  __int64 v27; // rcx
  int v28; // edx
  __int64 v29; // [rsp+8h] [rbp-188h]
  __int64 v30; // [rsp+10h] [rbp-180h]
  __int64 v32; // [rsp+20h] [rbp-170h]
  unsigned int v33; // [rsp+28h] [rbp-168h]
  int v34; // [rsp+2Ch] [rbp-164h]
  __int64 v35; // [rsp+30h] [rbp-160h] BYREF
  __int64 *v36; // [rsp+38h] [rbp-158h]
  __int64 *v37; // [rsp+40h] [rbp-150h]
  __int64 v38; // [rsp+48h] [rbp-148h]
  int v39; // [rsp+50h] [rbp-140h]
  _BYTE v40[312]; // [rsp+58h] [rbp-138h] BYREF

  v2 = (__int64 *)v40;
  v3 = *(_QWORD *)(a1 + 48);
  v4 = *(_QWORD *)(a1 + 40);
  v35 = 0;
  v36 = (__int64 *)v40;
  v37 = (__int64 *)v40;
  v38 = 32;
  v39 = 0;
  v30 = v3;
  v29 = v4;
  if ( v4 == v3 )
    return 0;
  v5 = (__int64 *)v40;
  while ( 1 )
  {
    v32 = *(_QWORD *)(v30 - 8);
    if ( v2 != v5 )
    {
LABEL_4:
      sub_16CCBA0((__int64)&v35, v32);
      v5 = v37;
      v2 = v36;
      goto LABEL_5;
    }
    v24 = &v2[HIDWORD(v38)];
    if ( v2 == v24 )
    {
LABEL_49:
      if ( HIDWORD(v38) >= (unsigned int)v38 )
        goto LABEL_4;
      ++HIDWORD(v38);
      *v24 = v32;
      v2 = v36;
      ++v35;
      v5 = v37;
    }
    else
    {
      v25 = v2;
      v26 = 0;
      while ( *(_QWORD *)(v30 - 8) != *v25 )
      {
        if ( *v25 == -2 )
          v26 = v25;
        if ( v24 == ++v25 )
        {
          if ( !v26 )
            goto LABEL_49;
          *v26 = v32;
          v5 = v37;
          --v39;
          v2 = v36;
          ++v35;
          break;
        }
      }
    }
LABEL_5:
    v6 = sub_157EBA0(v32);
    if ( v6 )
      break;
LABEL_21:
    v30 -= 8;
    if ( v29 == v30 )
    {
      v10 = (unsigned __int64)v5;
      v21 = 0;
      goto LABEL_26;
    }
  }
  v34 = sub_15F4D60(v6);
  v7 = sub_157EBA0(v32);
  if ( !v34 )
  {
    v5 = v37;
    v2 = v36;
    goto LABEL_21;
  }
  v8 = 0;
  v33 = ((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4);
  while ( 1 )
  {
    v9 = sub_15F4DF0(v7, v8);
    v10 = (unsigned __int64)v37;
    v11 = v9;
    v12 = v36;
    if ( v37 == v36 )
    {
      v13 = &v37[HIDWORD(v38)];
      if ( v37 == v13 )
      {
        v27 = (__int64)v37;
      }
      else
      {
        do
        {
          if ( v11 == *v12 )
            break;
          ++v12;
        }
        while ( v13 != v12 );
        v27 = (__int64)&v37[HIDWORD(v38)];
      }
    }
    else
    {
      v13 = &v37[(unsigned int)v38];
      v12 = sub_16CC9F0((__int64)&v35, v11);
      if ( v11 == *v12 )
      {
        v10 = (unsigned __int64)v37;
        v27 = (__int64)(v37 == v36 ? &v37[HIDWORD(v38)] : &v37[(unsigned int)v38]);
      }
      else
      {
        v10 = (unsigned __int64)v37;
        if ( v37 != v36 )
        {
          v12 = &v37[(unsigned int)v38];
          goto LABEL_12;
        }
        v12 = &v37[HIDWORD(v38)];
        v27 = (__int64)v12;
      }
    }
    while ( (__int64 *)v27 != v12 && (unsigned __int64)*v12 >= 0xFFFFFFFFFFFFFFFELL )
      ++v12;
LABEL_12:
    if ( v13 == v12 )
      goto LABEL_19;
    v14 = *(_DWORD *)(a2 + 24);
    if ( !v14 )
      goto LABEL_25;
    v15 = v14 - 1;
    v16 = *(_QWORD *)(a2 + 8);
    v17 = v15 & v33;
    v18 = (__int64 *)(v16 + 16LL * (v15 & v33));
    v19 = *v18;
    if ( v32 != *v18 )
      break;
LABEL_15:
    v20 = (__int64 **)v18[1];
    if ( !v20 )
      goto LABEL_25;
    while ( v11 != *v20[4] )
    {
      v20 = (__int64 **)*v20;
      if ( !v20 )
        goto LABEL_25;
    }
LABEL_19:
    if ( ++v8 == v34 )
    {
      v2 = v36;
      v5 = (__int64 *)v10;
      goto LABEL_21;
    }
  }
  v22 = 1;
  while ( v19 != -8 )
  {
    v28 = v22 + 1;
    v17 = v15 & (v22 + v17);
    v18 = (__int64 *)(v16 + 16LL * v17);
    v19 = *v18;
    if ( v32 == *v18 )
      goto LABEL_15;
    v22 = v28;
  }
LABEL_25:
  v2 = v36;
  v21 = 1;
LABEL_26:
  if ( v2 != (__int64 *)v10 )
    _libc_free(v10);
  return v21;
}
