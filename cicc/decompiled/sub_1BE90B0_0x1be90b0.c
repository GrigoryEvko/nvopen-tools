// Function: sub_1BE90B0
// Address: 0x1be90b0
//
__int64 __fastcall sub_1BE90B0(_QWORD *a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // rbx
  __int64 v6; // r12
  _QWORD *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // r8
  unsigned int v11; // r9d
  __int64 v12; // rax
  char v13; // dl
  __int64 *v14; // r15
  __int64 *v15; // rax
  __int64 *v16; // rdi
  __int64 *v17; // rsi
  __int64 *v18; // [rsp+8h] [rbp-2B8h]
  __int64 *v19; // [rsp+18h] [rbp-2A8h]
  unsigned int v20; // [rsp+24h] [rbp-29Ch]
  __int64 v21; // [rsp+28h] [rbp-298h] BYREF
  __int64 v22; // [rsp+30h] [rbp-290h] BYREF
  __int64 v23; // [rsp+38h] [rbp-288h] BYREF
  __int64 v24; // [rsp+40h] [rbp-280h] BYREF
  __int64 v25; // [rsp+48h] [rbp-278h] BYREF
  _QWORD *v26; // [rsp+50h] [rbp-270h] BYREF
  __int64 v27; // [rsp+58h] [rbp-268h]
  _QWORD v28[32]; // [rsp+60h] [rbp-260h] BYREF
  __int64 v29; // [rsp+160h] [rbp-160h] BYREF
  __int64 *v30; // [rsp+168h] [rbp-158h]
  __int64 *v31; // [rsp+170h] [rbp-150h]
  __int64 v32; // [rsp+178h] [rbp-148h]
  int v33; // [rsp+180h] [rbp-140h]
  _BYTE v34[312]; // [rsp+188h] [rbp-138h] BYREF

  v5 = (__int64)(a1 + 3);
  v21 = a2;
  v18 = sub_1BE8E40((__int64)(a1 + 3), &v21);
  if ( *((_DWORD *)v18 + 2) < a3 )
    return v21;
  v29 = 0;
  v26 = v28;
  v27 = 0x2000000000LL;
  v30 = (__int64 *)v34;
  v31 = (__int64 *)v34;
  v32 = 32;
  v33 = 0;
  if ( *((_DWORD *)v18 + 3) < a3 )
    return v18[3];
  LODWORD(v27) = 1;
  v8 = v28;
  v28[0] = v21;
  v9 = 1;
  while ( 1 )
  {
    v22 = v8[v9 - 1];
    v14 = sub_1BE8E40(v5, &v22);
    v10 = *(_QWORD *)(*a1 + 8LL * *((unsigned int *)v14 + 3));
    v11 = *((_DWORD *)v14 + 3);
    v15 = v30;
    v23 = v10;
    if ( v31 != v30 )
      break;
    v16 = &v30[HIDWORD(v32)];
    if ( v30 != v16 )
    {
      v17 = 0;
      do
      {
        if ( v10 == *v15 )
        {
          LODWORD(v12) = v27;
          goto LABEL_19;
        }
        if ( *v15 == -2 )
          v17 = v15;
        ++v15;
      }
      while ( v16 != v15 );
      if ( v17 )
      {
        *v17 = v10;
        v12 = (unsigned int)v27;
        --v33;
        v11 = *((_DWORD *)v14 + 3);
        ++v29;
        if ( a3 > v11 )
        {
LABEL_8:
          v9 = (unsigned int)(v12 - 1);
          LODWORD(v27) = v9;
          goto LABEL_9;
        }
        goto LABEL_29;
      }
    }
    if ( HIDWORD(v32) >= (unsigned int)v32 )
      break;
    ++HIDWORD(v32);
    *v16 = v10;
    v11 = *((_DWORD *)v14 + 3);
    ++v29;
    v12 = (unsigned int)v27;
LABEL_7:
    if ( a3 > v11 )
      goto LABEL_8;
LABEL_29:
    if ( (unsigned int)v12 >= HIDWORD(v27) )
    {
      sub_16CD150((__int64)&v26, v28, 0, 8, v10, v11);
      v12 = (unsigned int)v27;
    }
    v26[v12] = v23;
    v9 = (unsigned int)(v27 + 1);
    LODWORD(v27) = v27 + 1;
LABEL_9:
    if ( !(_DWORD)v9 )
      goto LABEL_23;
LABEL_10:
    v8 = v26;
  }
  sub_16CCBA0((__int64)&v29, v10);
  v11 = *((_DWORD *)v14 + 3);
  v12 = (unsigned int)v27;
  if ( v13 )
    goto LABEL_7;
LABEL_19:
  v9 = (unsigned int)(v12 - 1);
  LODWORD(v27) = v9;
  if ( a3 > v11 )
    goto LABEL_9;
  v19 = sub_1BE8E40(v5, &v23);
  v24 = v19[3];
  v25 = v14[3];
  v20 = *((_DWORD *)sub_1BE8E40(v5, &v24) + 4);
  if ( v20 < *((_DWORD *)sub_1BE8E40(v5, &v25) + 4) )
    v14[3] = v24;
  *((_DWORD *)v14 + 3) = *((_DWORD *)v19 + 3);
  v9 = (unsigned int)v27;
  if ( (_DWORD)v27 )
    goto LABEL_10;
LABEL_23:
  v6 = v18[3];
  if ( v31 != v30 )
    _libc_free((unsigned __int64)v31);
  if ( v26 != v28 )
    _libc_free((unsigned __int64)v26);
  return v6;
}
