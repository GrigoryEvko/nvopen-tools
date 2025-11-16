// Function: sub_29DCFA0
// Address: 0x29dcfa0
//
__int64 __fastcall sub_29DCFA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  unsigned int i; // eax
  unsigned __int8 **v8; // rdx
  unsigned __int8 *v9; // rbx
  unsigned __int8 **v10; // rax
  char v11; // dl
  int v12; // eax
  unsigned int v13; // r12d
  __int64 v15; // rbx
  unsigned __int8 *v16; // r15
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  unsigned __int8 **v19; // [rsp+0h] [rbp-E0h] BYREF
  __int64 v20; // [rsp+8h] [rbp-D8h]
  _QWORD v21[8]; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v22; // [rsp+50h] [rbp-90h] BYREF
  unsigned __int8 **v23; // [rsp+58h] [rbp-88h]
  __int64 v24; // [rsp+60h] [rbp-80h]
  int v25; // [rsp+68h] [rbp-78h]
  unsigned __int8 v26; // [rsp+6Ch] [rbp-74h]
  char v27; // [rsp+70h] [rbp-70h] BYREF

  v6 = 1;
  v19 = (unsigned __int8 **)v21;
  v22 = 0;
  v24 = 8;
  v25 = 0;
  v26 = 1;
  v21[0] = a1;
  v23 = (unsigned __int8 **)&v27;
  v20 = 0x800000001LL;
  for ( i = 1; ; i = v20 )
  {
LABEL_2:
    if ( !i )
    {
      v13 = 1;
      goto LABEL_13;
    }
    v8 = v19;
    v9 = v19[i - 1];
    LODWORD(v20) = i - 1;
    if ( !(_BYTE)v6 )
      goto LABEL_10;
    v10 = v23;
    v8 = &v23[HIDWORD(v24)];
    if ( v23 != v8 )
    {
      while ( v9 != *v10 )
      {
        if ( v8 == ++v10 )
          goto LABEL_9;
      }
      continue;
    }
LABEL_9:
    if ( HIDWORD(v24) < (unsigned int)v24 )
    {
      ++HIDWORD(v24);
      *v8 = v9;
      v6 = v26;
      ++v22;
    }
    else
    {
LABEL_10:
      sub_C8CC70((__int64)&v22, (__int64)v9, (__int64)v8, v6, a5, a6);
      v6 = v26;
      if ( !v11 )
        continue;
    }
    v12 = *v9;
    if ( (unsigned __int8)v12 <= 3u || (unsigned int)(v12 - 12) <= 9 )
    {
      v13 = 0;
LABEL_13:
      if ( (_BYTE)v6 )
        goto LABEL_14;
LABEL_25:
      _libc_free((unsigned __int64)v23);
      goto LABEL_14;
    }
    v15 = *((_QWORD *)v9 + 2);
    if ( v15 )
      break;
  }
  while ( 1 )
  {
    v16 = *(unsigned __int8 **)(v15 + 24);
    if ( *v16 > 0x15u )
      break;
    v17 = (unsigned int)v20;
    v18 = (unsigned int)v20 + 1LL;
    if ( v18 > HIDWORD(v20) )
    {
      sub_C8D5F0((__int64)&v19, v21, v18, 8u, a5, a6);
      v17 = (unsigned int)v20;
    }
    v19[v17] = v16;
    i = v20 + 1;
    LODWORD(v20) = v20 + 1;
    v15 = *(_QWORD *)(v15 + 8);
    if ( !v15 )
    {
      v6 = v26;
      goto LABEL_2;
    }
  }
  v13 = 0;
  if ( !v26 )
    goto LABEL_25;
LABEL_14:
  if ( v19 != v21 )
    _libc_free((unsigned __int64)v19);
  return v13;
}
