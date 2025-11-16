// Function: sub_2B1D420
// Address: 0x2b1d420
//
__int64 __fastcall sub_2B1D420(
        unsigned __int8 *a1,
        unsigned __int8 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned int v8; // ebx
  char i; // al
  __int64 v10; // rcx
  unsigned __int8 **v11; // rdx
  unsigned __int8 **j; // rax
  unsigned int v13; // r12d
  unsigned __int8 **v14; // rax
  unsigned __int8 **v15; // rdx
  unsigned __int8 **v16; // rax
  unsigned __int8 **v17; // rdx
  unsigned __int8 **v19; // rax
  unsigned __int8 **v20; // rax
  __int64 v21; // [rsp+0h] [rbp-160h] BYREF
  unsigned __int8 **v22; // [rsp+8h] [rbp-158h]
  __int64 v23; // [rsp+10h] [rbp-150h]
  int v24; // [rsp+18h] [rbp-148h]
  char v25; // [rsp+1Ch] [rbp-144h]
  char v26; // [rsp+20h] [rbp-140h] BYREF
  __int64 v27; // [rsp+A0h] [rbp-C0h] BYREF
  unsigned __int8 **v28; // [rsp+A8h] [rbp-B8h]
  __int64 v29; // [rsp+B0h] [rbp-B0h]
  int v30; // [rsp+B8h] [rbp-A8h]
  unsigned __int8 v31; // [rsp+BCh] [rbp-A4h]
  char v32; // [rsp+C0h] [rbp-A0h] BYREF

  v8 = 0;
  v25 = 1;
  v21 = 0;
  v23 = 16;
  v24 = 0;
  v27 = 0;
  v29 = 16;
  v30 = 0;
  v31 = 1;
  v22 = (unsigned __int8 **)&v26;
  v28 = (unsigned __int8 **)&v32;
  for ( i = 1; ; i = v25 )
  {
    if ( i )
    {
      v10 = (__int64)v22;
      v11 = &v22[HIDWORD(v23)];
      for ( j = v22; v11 != j; ++j )
      {
        if ( a2 == *j )
          goto LABEL_9;
      }
    }
    else if ( sub_C8CA60((__int64)&v21, (__int64)a2) )
    {
      goto LABEL_22;
    }
    if ( v31 )
      break;
    if ( sub_C8CA60((__int64)&v27, (__int64)a1) )
      goto LABEL_22;
LABEL_31:
    if ( a2 == a1 || (unsigned int)qword_500FC48 < v8 )
      goto LABEL_24;
    if ( !v25 )
    {
LABEL_45:
      sub_C8CC70((__int64)&v21, (__int64)a1, (__int64)v17, v10, a5, a6);
      goto LABEL_38;
    }
    v19 = v22;
    v10 = HIDWORD(v23);
    v17 = &v22[HIDWORD(v23)];
    if ( v22 == v17 )
    {
LABEL_48:
      if ( HIDWORD(v23) >= (unsigned int)v23 )
        goto LABEL_45;
      v10 = (unsigned int)++HIDWORD(v23);
      *v17 = a1;
      ++v21;
    }
    else
    {
      while ( a1 != *v19 )
      {
        if ( v17 == ++v19 )
          goto LABEL_48;
      }
    }
LABEL_38:
    if ( !v31 )
      goto LABEL_44;
    v20 = v28;
    v10 = HIDWORD(v29);
    v17 = &v28[HIDWORD(v29)];
    if ( v28 == v17 )
    {
LABEL_46:
      if ( HIDWORD(v29) >= (unsigned int)v29 )
      {
LABEL_44:
        sub_C8CC70((__int64)&v27, (__int64)a2, (__int64)v17, v10, a5, a6);
        goto LABEL_43;
      }
      ++HIDWORD(v29);
      *v17 = a2;
      ++v27;
    }
    else
    {
      while ( a2 != *v20 )
      {
        if ( v17 == ++v20 )
          goto LABEL_46;
      }
    }
LABEL_43:
    ++v8;
    a1 = sub_98ACB0(a1, 1u);
    a2 = sub_98ACB0(a2, 1u);
  }
  v16 = v28;
  v17 = &v28[HIDWORD(v29)];
  if ( v28 == v17 )
    goto LABEL_31;
  while ( a1 != *v16 )
  {
    if ( v17 == ++v16 )
      goto LABEL_31;
  }
LABEL_22:
  if ( v25 )
  {
    v10 = (__int64)v22;
    v11 = &v22[HIDWORD(v23)];
    if ( v11 != v22 )
    {
LABEL_9:
      while ( a2 != *(unsigned __int8 **)v10 )
      {
        v10 += 8;
        if ( (unsigned __int8 **)v10 == v11 )
          goto LABEL_24;
      }
      goto LABEL_10;
    }
LABEL_24:
    v13 = 0;
LABEL_25:
    if ( !v31 )
      _libc_free((unsigned __int64)v28);
    goto LABEL_27;
  }
  if ( !sub_C8CA60((__int64)&v21, (__int64)a2) )
    goto LABEL_24;
LABEL_10:
  v13 = v31;
  if ( !v31 )
  {
    LOBYTE(v13) = sub_C8CA60((__int64)&v27, (__int64)a1) == 0;
    goto LABEL_25;
  }
  v14 = v28;
  v15 = &v28[HIDWORD(v29)];
  if ( v28 != v15 )
  {
    while ( a1 != *v14 )
    {
      if ( v15 == ++v14 )
        goto LABEL_27;
    }
    v13 = 0;
  }
LABEL_27:
  if ( !v25 )
    _libc_free((unsigned __int64)v22);
  return v13;
}
