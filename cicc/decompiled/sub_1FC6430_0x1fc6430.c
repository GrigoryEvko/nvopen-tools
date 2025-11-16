// Function: sub_1FC6430
// Address: 0x1fc6430
//
__int64 __fastcall sub_1FC6430(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5, _QWORD *a6)
{
  __int64 result; // rax
  bool v7; // zf
  unsigned __int64 *v8; // rax
  _QWORD *v10; // r9
  int v11; // eax
  _QWORD *v12; // r8
  int v13; // edi
  unsigned int v14; // edx
  __int64 v15; // rcx
  __int64 v16; // rsi
  __int64 *v17; // rax
  __int64 *v18; // r12
  __int64 *v19; // r15
  __int64 v20; // rdx
  unsigned int v21; // edx
  unsigned int v22; // r10d
  __int64 v23; // [rsp+18h] [rbp-168h] BYREF
  __int64 v24; // [rsp+28h] [rbp-158h] BYREF
  __int64 v25; // [rsp+30h] [rbp-150h] BYREF
  __int64 v26; // [rsp+38h] [rbp-148h]
  _QWORD *v27; // [rsp+40h] [rbp-140h] BYREF
  int v28; // [rsp+48h] [rbp-138h]
  _BYTE *v29; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v30; // [rsp+C8h] [rbp-B8h]
  _BYTE v31[176]; // [rsp+D0h] [rbp-B0h] BYREF

  result = 0;
  v7 = *(_QWORD *)(a2 + 48) == 0;
  v23 = a2;
  if ( !v7 )
    return result;
  v8 = (unsigned __int64 *)&v27;
  v25 = 0;
  v26 = 1;
  do
    *v8++ = -8;
  while ( v8 != (unsigned __int64 *)&v29 );
  v29 = v31;
  v30 = 0x1000000000LL;
  sub_1FC6130((__int64)&v25, &v23, (__int64)&v29, a4, a5, a6);
  v11 = v30;
  do
  {
    while ( 1 )
    {
      v15 = (__int64)v29;
      v16 = *(_QWORD *)&v29[8 * v11 - 8];
      if ( (v26 & 1) != 0 )
      {
        v12 = &v27;
        v13 = 15;
      }
      else
      {
        v12 = v27;
        v13 = v28 - 1;
        if ( !v28 )
          goto LABEL_9;
      }
      v14 = v13 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v15 = (__int64)&v12[v14];
      v10 = *(_QWORD **)v15;
      if ( *(_QWORD *)v15 == v16 )
      {
LABEL_8:
        *(_QWORD *)v15 = -16;
        ++HIDWORD(v26);
        LODWORD(v26) = (2 * ((unsigned int)v26 >> 1) - 2) | v26 & 1;
        v11 = v30;
      }
      else
      {
        v15 = 1;
        while ( v10 != (_QWORD *)-8LL )
        {
          v22 = v15 + 1;
          v14 = v13 & (v15 + v14);
          v15 = (__int64)&v12[v14];
          v10 = *(_QWORD **)v15;
          if ( v16 == *(_QWORD *)v15 )
            goto LABEL_8;
          v15 = v22;
        }
      }
LABEL_9:
      --v11;
      v23 = v16;
      LODWORD(v30) = v11;
      if ( v16 )
        break;
LABEL_12:
      if ( !v11 )
        goto LABEL_21;
    }
    if ( *(_QWORD *)(v16 + 48) )
    {
      sub_1F81BC0((__int64)a1, v16);
      v11 = v30;
      goto LABEL_12;
    }
    v17 = *(__int64 **)(v16 + 32);
    v18 = &v17[5 * *(unsigned int *)(v16 + 56)];
    if ( v17 != v18 )
    {
      v19 = *(__int64 **)(v16 + 32);
      do
      {
        v20 = *v19;
        v19 += 5;
        v24 = v20;
        sub_1FC6130((__int64)&v25, &v24, v20, v15, v12, v10);
      }
      while ( v18 != v19 );
      v16 = v23;
    }
    sub_1F6D2A0((__int64)a1, v16);
    sub_1D2DE10(*a1, v23, v21);
    v11 = v30;
  }
  while ( (_DWORD)v30 );
LABEL_21:
  if ( v29 != v31 )
    _libc_free((unsigned __int64)v29);
  if ( (v26 & 1) == 0 )
    j___libc_free_0(v27);
  return 1;
}
