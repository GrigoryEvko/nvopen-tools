// Function: sub_3962BB0
// Address: 0x3962bb0
//
__int64 __fastcall sub_3962BB0(__int64 **a1, __int64 a2)
{
  __int64 *v2; // rsi
  __int64 *v3; // rax
  __int64 v4; // rbx
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // r13
  unsigned int v7; // r14d
  __int64 v8; // rax
  unsigned __int64 v9; // rdi
  __int64 v10; // rbx
  _QWORD *v11; // rax
  _BYTE *v12; // r12
  int v13; // ecx
  int v14; // ecx
  __int64 v15; // r9
  unsigned int v16; // esi
  __int64 *v17; // rax
  __int64 v18; // r11
  __int64 **v19; // rax
  unsigned int v20; // r12d
  int v21; // eax
  __int64 v23; // rcx
  int v24; // edx
  __int64 *v25; // [rsp+0h] [rbp-190h]
  __int64 *v26; // [rsp+8h] [rbp-188h]
  __int64 *v27; // [rsp+10h] [rbp-180h]
  __int64 v28; // [rsp+18h] [rbp-178h]
  unsigned int v30; // [rsp+28h] [rbp-168h]
  int v31; // [rsp+2Ch] [rbp-164h]
  __int64 v32; // [rsp+30h] [rbp-160h] BYREF
  _BYTE *v33; // [rsp+38h] [rbp-158h]
  _BYTE *v34; // [rsp+40h] [rbp-150h]
  __int64 v35; // [rsp+48h] [rbp-148h]
  int v36; // [rsp+50h] [rbp-140h]
  _BYTE v37[312]; // [rsp+58h] [rbp-138h] BYREF

  v2 = *a1;
  v33 = v37;
  v34 = v37;
  v3 = a1[1];
  v32 = 0;
  v35 = 32;
  v36 = 0;
  v25 = v2;
  if ( v2 == v3 )
    return 0;
  v27 = v3 - 1;
  while ( 1 )
  {
    v26 = v27;
    v4 = *v27;
    v28 = *v27;
    sub_1412190((__int64)&v32, *v27);
    v5 = sub_157EBA0(v4);
    if ( v5 )
    {
      v31 = sub_15F4D60(v5);
      v6 = sub_157EBA0(v28);
      if ( v31 )
        break;
    }
LABEL_18:
    --v27;
    if ( v25 == v26 )
    {
      v9 = (unsigned __int64)v34;
      v20 = 0;
      goto LABEL_23;
    }
  }
  v7 = 0;
  v30 = ((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4);
  while ( 1 )
  {
    v8 = sub_15F4DF0(v6, v7);
    v9 = (unsigned __int64)v34;
    v10 = v8;
    v11 = v33;
    if ( v34 == v33 )
    {
      v12 = &v34[8 * HIDWORD(v35)];
      if ( v34 == v12 )
      {
        v23 = (__int64)v34;
      }
      else
      {
        do
        {
          if ( v10 == *v11 )
            break;
          ++v11;
        }
        while ( v12 != (_BYTE *)v11 );
        v23 = (__int64)&v34[8 * HIDWORD(v35)];
      }
    }
    else
    {
      v12 = &v34[8 * (unsigned int)v35];
      v11 = sub_16CC9F0((__int64)&v32, v10);
      if ( v10 == *v11 )
      {
        v9 = (unsigned __int64)v34;
        v23 = (__int64)(v34 == v33 ? &v34[8 * HIDWORD(v35)] : &v34[8 * (unsigned int)v35]);
      }
      else
      {
        v9 = (unsigned __int64)v34;
        if ( v34 != v33 )
        {
          v11 = &v34[8 * (unsigned int)v35];
          goto LABEL_10;
        }
        v11 = &v34[8 * HIDWORD(v35)];
        v23 = (__int64)v11;
      }
    }
    while ( (_QWORD *)v23 != v11 && *v11 >= 0xFFFFFFFFFFFFFFFELL )
      ++v11;
LABEL_10:
    if ( v12 == (_BYTE *)v11 )
      goto LABEL_17;
    v13 = *(_DWORD *)(a2 + 24);
    if ( !v13 )
      goto LABEL_22;
    v14 = v13 - 1;
    v15 = *(_QWORD *)(a2 + 8);
    v16 = v14 & v30;
    v17 = (__int64 *)(v15 + 16LL * (v14 & v30));
    v18 = *v17;
    if ( v28 != *v17 )
      break;
LABEL_13:
    v19 = (__int64 **)v17[1];
    if ( !v19 )
      goto LABEL_22;
    while ( v10 != *v19[4] )
    {
      v19 = (__int64 **)*v19;
      if ( !v19 )
        goto LABEL_22;
    }
LABEL_17:
    if ( v31 == ++v7 )
      goto LABEL_18;
  }
  v21 = 1;
  while ( v18 != -8 )
  {
    v24 = v21 + 1;
    v16 = v14 & (v21 + v16);
    v17 = (__int64 *)(v15 + 16LL * v16);
    v18 = *v17;
    if ( v28 == *v17 )
      goto LABEL_13;
    v21 = v24;
  }
LABEL_22:
  v20 = 1;
LABEL_23:
  if ( (_BYTE *)v9 != v33 )
    _libc_free(v9);
  return v20;
}
