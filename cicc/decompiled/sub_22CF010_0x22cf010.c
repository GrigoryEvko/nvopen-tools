// Function: sub_22CF010
// Address: 0x22cf010
//
__int64 __fastcall sub_22CF010(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v10; // dl
  unsigned int v11; // eax
  unsigned __int8 v12; // al
  unsigned int v14; // eax
  unsigned int v15; // eax
  unsigned int v16; // eax
  unsigned int v17; // eax
  unsigned int v18; // eax
  unsigned int v19; // eax
  bool v20; // zf
  unsigned __int8 v21; // [rsp+20h] [rbp-90h] BYREF
  char v22; // [rsp+21h] [rbp-8Fh]
  unsigned __int64 v23; // [rsp+28h] [rbp-88h] BYREF
  unsigned int v24; // [rsp+30h] [rbp-80h]
  unsigned __int64 v25; // [rsp+38h] [rbp-78h] BYREF
  unsigned int v26; // [rsp+40h] [rbp-70h]
  char v27; // [rsp+48h] [rbp-68h]
  unsigned __int8 v28; // [rsp+50h] [rbp-60h] BYREF
  char v29; // [rsp+51h] [rbp-5Fh]
  unsigned __int64 v30; // [rsp+58h] [rbp-58h]
  unsigned int v31; // [rsp+60h] [rbp-50h]
  unsigned __int64 v32; // [rsp+68h] [rbp-48h]
  unsigned int v33; // [rsp+70h] [rbp-40h]
  char v34; // [rsp+78h] [rbp-38h]

  sub_22CCF60((__int64)&v21, a2, a3, a4, a5, a6);
  while ( !v27 )
  {
    while ( 1 )
    {
      sub_22CDAD0(a2);
      sub_22CCF60((__int64)&v28, a2, a3, a4, a5, a6);
      if ( v27 )
        break;
      if ( v34 )
      {
        v22 = 0;
        v21 = v28;
        if ( v28 > 3u )
        {
          if ( (unsigned __int8)(v28 - 4) <= 1u )
          {
            v14 = v31;
            v31 = 0;
            v24 = v14;
            v23 = v30;
            v15 = v33;
            v33 = 0;
            v26 = v15;
            v25 = v32;
            v22 = v29;
          }
        }
        else if ( v28 > 1u )
        {
          v23 = v30;
        }
        v28 = 0;
        v27 = 1;
LABEL_16:
        v34 = 0;
        if ( (unsigned int)v28 - 4 <= 1 )
        {
          if ( v33 > 0x40 && v32 )
            j_j___libc_free_0_0(v32);
          if ( v31 > 0x40 && v30 )
            j_j___libc_free_0_0(v30);
        }
        goto LABEL_5;
      }
    }
    v10 = v34;
    v11 = v21 - 4;
    if ( v34 )
    {
      if ( v11 <= 1 )
      {
        if ( v26 > 0x40 && v25 )
          j_j___libc_free_0_0(v25);
        if ( v24 > 0x40 && v23 )
          j_j___libc_free_0_0(v23);
        v10 = v34;
      }
      v22 = 0;
      v21 = v28;
      if ( v28 > 3u )
      {
        if ( (unsigned __int8)(v28 - 4) <= 1u )
        {
          v16 = v31;
          v31 = 0;
          v24 = v16;
          v23 = v30;
          v17 = v33;
          v33 = 0;
          v26 = v17;
          v25 = v32;
          v22 = v29;
        }
      }
      else if ( v28 > 1u )
      {
        v23 = v30;
      }
      v28 = 0;
    }
    else
    {
      v27 = 0;
      if ( v11 > 1 )
        continue;
      if ( v26 > 0x40 && v25 )
        j_j___libc_free_0_0(v25);
      if ( v24 > 0x40 && v23 )
        j_j___libc_free_0_0(v23);
      v10 = v34;
    }
    if ( v10 )
      goto LABEL_16;
LABEL_5:
    ;
  }
  v12 = v21;
  *(_WORD *)a1 = v21;
  if ( v12 <= 3u )
  {
    if ( v12 > 1u )
      *(_QWORD *)(a1 + 8) = v23;
LABEL_9:
    if ( !v27 )
      return a1;
LABEL_40:
    v27 = 0;
    sub_22C0090(&v21);
    return a1;
  }
  if ( (unsigned __int8)(v12 - 4) > 1u )
    goto LABEL_9;
  v18 = v24;
  *(_DWORD *)(a1 + 16) = v24;
  if ( v18 > 0x40 )
    sub_C43780(a1 + 8, (const void **)&v23);
  else
    *(_QWORD *)(a1 + 8) = v23;
  v19 = v26;
  *(_DWORD *)(a1 + 32) = v26;
  if ( v19 > 0x40 )
    sub_C43780(a1 + 24, (const void **)&v25);
  else
    *(_QWORD *)(a1 + 24) = v25;
  v20 = v27 == 0;
  *(_BYTE *)(a1 + 1) = v22;
  if ( !v20 )
    goto LABEL_40;
  return a1;
}
