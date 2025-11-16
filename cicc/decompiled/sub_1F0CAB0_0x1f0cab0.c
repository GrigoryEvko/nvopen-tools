// Function: sub_1F0CAB0
// Address: 0x1f0cab0
//
__int64 __fastcall sub_1F0CAB0(__int64 *a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 v4; // rcx
  __int64 v5; // rsi
  __int64 *v6; // rdi
  __int64 *v7; // r15
  __int64 v8; // rbx
  __int64 *v9; // r13
  int v10; // edx
  int v11; // edx
  __int64 v12; // r9
  unsigned int v13; // esi
  __int64 *v14; // rax
  __int64 v15; // r11
  __int64 **v16; // rax
  unsigned int v17; // r12d
  int v18; // eax
  __int64 *v20; // r8
  __int64 *v21; // rdx
  __int64 *v22; // rsi
  __int64 v23; // rdx
  int v24; // ecx
  __int64 v25; // [rsp+8h] [rbp-188h]
  __int64 v26; // [rsp+10h] [rbp-180h]
  unsigned int v27; // [rsp+1Ch] [rbp-174h]
  __int64 v28; // [rsp+20h] [rbp-170h]
  __int64 *v29; // [rsp+28h] [rbp-168h]
  __int64 v30; // [rsp+30h] [rbp-160h] BYREF
  __int64 *v31; // [rsp+38h] [rbp-158h]
  __int64 *v32; // [rsp+40h] [rbp-150h]
  __int64 v33; // [rsp+48h] [rbp-148h]
  int v34; // [rsp+50h] [rbp-140h]
  _BYTE v35[312]; // [rsp+58h] [rbp-138h] BYREF

  v2 = (__int64 *)v35;
  v4 = a1[1];
  v5 = *a1;
  v30 = 0;
  v31 = (__int64 *)v35;
  v32 = (__int64 *)v35;
  v33 = 32;
  v34 = 0;
  v26 = v4;
  v25 = v5;
  if ( v4 == v5 )
    return 0;
  v6 = (__int64 *)v35;
  while ( 1 )
  {
    v28 = *(_QWORD *)(v26 - 8);
    if ( v2 != v6 )
    {
LABEL_4:
      sub_16CCBA0((__int64)&v30, v28);
      v6 = v32;
      v2 = v31;
      goto LABEL_5;
    }
    v20 = &v2[HIDWORD(v33)];
    if ( v2 == v20 )
    {
LABEL_47:
      if ( HIDWORD(v33) >= (unsigned int)v33 )
        goto LABEL_4;
      ++HIDWORD(v33);
      *v20 = v28;
      v2 = v31;
      ++v30;
      v6 = v32;
    }
    else
    {
      v21 = v2;
      v22 = 0;
      while ( *(_QWORD *)(v26 - 8) != *v21 )
      {
        if ( *v21 == -2 )
          v22 = v21;
        if ( v20 == ++v21 )
        {
          if ( !v22 )
            goto LABEL_47;
          *v22 = v28;
          v6 = v32;
          --v34;
          v2 = v31;
          ++v30;
          break;
        }
      }
    }
LABEL_5:
    v29 = *(__int64 **)(v28 + 96);
    if ( *(__int64 **)(v28 + 88) != v29 )
      break;
LABEL_19:
    v26 -= 8;
    if ( v25 == v26 )
    {
      v17 = 0;
      goto LABEL_24;
    }
  }
  v27 = ((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4);
  v7 = *(__int64 **)(v28 + 88);
  while ( 1 )
  {
    v8 = *v7;
    if ( v2 == v6 )
    {
      v9 = &v2[HIDWORD(v33)];
      if ( v2 == v9 )
      {
        v6 = v32;
        v23 = (__int64)v2;
      }
      else
      {
        do
        {
          if ( v8 == *v2 )
            break;
          ++v2;
        }
        while ( v9 != v2 );
        v6 = v32;
        v23 = (__int64)v9;
      }
    }
    else
    {
      v9 = &v6[(unsigned int)v33];
      v2 = sub_16CC9F0((__int64)&v30, *v7);
      if ( v8 == *v2 )
      {
        v6 = v32;
        v23 = (__int64)(v32 == v31 ? &v32[HIDWORD(v33)] : &v32[(unsigned int)v33]);
      }
      else
      {
        v6 = v32;
        if ( v32 != v31 )
        {
          v2 = &v32[(unsigned int)v33];
          goto LABEL_11;
        }
        v2 = &v32[HIDWORD(v33)];
        v23 = (__int64)v2;
      }
    }
    while ( (__int64 *)v23 != v2 && (unsigned __int64)*v2 >= 0xFFFFFFFFFFFFFFFELL )
      ++v2;
LABEL_11:
    if ( v9 == v2 )
      goto LABEL_18;
    v10 = *(_DWORD *)(a2 + 256);
    if ( !v10 )
      goto LABEL_23;
    v11 = v10 - 1;
    v12 = *(_QWORD *)(a2 + 240);
    v13 = v11 & v27;
    v14 = (__int64 *)(v12 + 16LL * (v11 & v27));
    v15 = *v14;
    if ( v28 != *v14 )
      break;
LABEL_14:
    v16 = (__int64 **)v14[1];
    if ( !v16 )
      goto LABEL_23;
    while ( v8 != *v16[4] )
    {
      v16 = (__int64 **)*v16;
      if ( !v16 )
        goto LABEL_23;
    }
LABEL_18:
    v2 = v31;
    if ( v29 == ++v7 )
      goto LABEL_19;
  }
  v18 = 1;
  while ( v15 != -8 )
  {
    v24 = v18 + 1;
    v13 = v11 & (v18 + v13);
    v14 = (__int64 *)(v12 + 16LL * v13);
    v15 = *v14;
    if ( v28 == *v14 )
      goto LABEL_14;
    v18 = v24;
  }
LABEL_23:
  v2 = v31;
  v17 = 1;
LABEL_24:
  if ( v2 != v6 )
    _libc_free((unsigned __int64)v6);
  return v17;
}
