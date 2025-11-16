// Function: sub_D9B790
// Address: 0xd9b790
//
__int64 __fastcall sub_D9B790(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r13d
  char *v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rdx
  char v14; // cl
  char **v16; // rax
  char **v17; // r13
  char **v18; // r12
  char **v19; // rax
  char **v20; // rdx
  char **v21; // rax
  __int64 *v22; // rax
  char v23[8]; // [rsp+0h] [rbp-C0h] BYREF
  __int64 v24; // [rsp+8h] [rbp-B8h]
  char **v25; // [rsp+10h] [rbp-B0h]
  __int64 v26; // [rsp+18h] [rbp-A8h]
  int v27; // [rsp+20h] [rbp-A0h]
  char v28; // [rsp+24h] [rbp-9Ch]
  char v29; // [rsp+28h] [rbp-98h] BYREF
  char v30[8]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v31; // [rsp+58h] [rbp-68h] BYREF
  char **v32; // [rsp+60h] [rbp-60h]
  __int64 v33; // [rsp+68h] [rbp-58h]
  int v34; // [rsp+70h] [rbp-50h]
  char v35; // [rsp+74h] [rbp-4Ch]
  char v36; // [rsp+78h] [rbp-48h] BYREF

  v6 = 1;
  v8 = v23;
  v25 = (char **)&v29;
  v23[0] = 1;
  v24 = 0;
  v26 = 4;
  v27 = 0;
  v28 = 1;
  sub_D9B3F0(a1, (__int64)v23, a3, a4, a5, a6);
  if ( HIDWORD(v26) == v27 )
    goto LABEL_6;
  v8 = v30;
  v30[0] = 0;
  v32 = (char **)&v36;
  v31 = 0;
  v33 = 4;
  v34 = 0;
  v35 = 1;
  sub_D9B3F0(a2, (__int64)v30, v9, v10, v11, v12);
  v13 = HIDWORD(v26);
  if ( HIDWORD(v26) - v27 > (unsigned int)(HIDWORD(v33) - v34) )
  {
    v14 = v35;
    v6 = 0;
    goto LABEL_4;
  }
  v16 = v25;
  if ( !v28 )
    v13 = (unsigned int)v26;
  v17 = &v25[v13];
  v14 = v35;
  if ( v25 == v17 )
    goto LABEL_14;
  while ( 1 )
  {
    v8 = *v16;
    v18 = v16;
    if ( (unsigned __int64)*v16 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v17 == ++v16 )
      goto LABEL_14;
  }
  if ( v17 == v16 )
    goto LABEL_14;
  if ( !v35 )
    goto LABEL_27;
LABEL_17:
  v19 = v32;
  v20 = &v32[HIDWORD(v33)];
  if ( v32 != v20 )
  {
    while ( *v19 != v8 )
    {
      if ( v20 == ++v19 )
        goto LABEL_28;
    }
    while ( 1 )
    {
      v21 = v18 + 1;
      if ( v18 + 1 == v17 )
        break;
      v8 = *v21;
      for ( ++v18; (unsigned __int64)*v21 >= 0xFFFFFFFFFFFFFFFELL; v18 = v21 )
      {
        if ( v17 == ++v21 )
          goto LABEL_14;
        v8 = *v21;
      }
      if ( v17 == v18 )
        break;
      if ( v14 )
        goto LABEL_17;
LABEL_27:
      v22 = sub_C8CA60((__int64)&v31, (__int64)v8);
      v14 = v35;
      if ( !v22 )
        goto LABEL_28;
    }
LABEL_14:
    v6 = 1;
    goto LABEL_4;
  }
LABEL_28:
  v6 = 0;
LABEL_4:
  if ( !v14 )
    _libc_free(v32, v8);
LABEL_6:
  if ( !v28 )
    _libc_free(v25, v8);
  return v6;
}
