// Function: sub_F61E50
// Address: 0xf61e50
//
__int64 __fastcall sub_F61E50(__int64 a1, __int64 *a2)
{
  __int64 v3; // rax
  __int64 v4; // r8
  __int64 v5; // r15
  __int64 v6; // rbx
  unsigned int v7; // r12d
  unsigned __int64 v8; // r8
  int v9; // eax
  __int64 v10; // rdi
  bool v11; // zf
  _QWORD *v12; // rax
  __int64 v13; // rdx
  _QWORD *v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // rdx
  _QWORD *v17; // rdx
  unsigned int v18; // eax
  int v19; // r10d
  int v20; // eax
  __int64 i; // rax
  __int64 v22; // rdi
  unsigned int v23; // eax
  __int64 *v24; // rcx
  __int64 v25; // r8
  int v27; // ecx
  int v28; // r10d
  unsigned __int64 v29; // [rsp+8h] [rbp-E8h]
  __int64 v30; // [rsp+10h] [rbp-E0h] BYREF
  __int64 v31; // [rsp+18h] [rbp-D8h]
  __int64 v32; // [rsp+20h] [rbp-D0h]
  __int64 v33; // [rsp+28h] [rbp-C8h]
  _BYTE *v34; // [rsp+30h] [rbp-C0h]
  __int64 v35; // [rsp+38h] [rbp-B8h]
  _BYTE v36[176]; // [rsp+40h] [rbp-B0h] BYREF

  v3 = sub_AA4E30(a1);
  v4 = *(_QWORD *)(a1 + 48);
  v5 = *(_QWORD *)(a1 + 56);
  v6 = v3;
  v7 = 0;
  v30 = 0;
  v8 = v4 & 0xFFFFFFFFFFFFFFF8LL;
  v34 = v36;
  v35 = 0x1000000000LL;
  v9 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  if ( v5 == v8 )
    goto LABEL_27;
  while ( 1 )
  {
    v10 = v5 - 24;
    v11 = v5 == 0;
    v5 = *(_QWORD *)(v5 + 8);
    if ( v11 )
      v10 = 0;
    if ( !v9 )
    {
      v12 = v34;
      v13 = 8LL * (unsigned int)v35;
      v14 = &v34[v13];
      v15 = v13 >> 3;
      v16 = v13 >> 5;
      if ( v16 )
      {
        v17 = &v34[32 * v16];
        while ( v10 != *v12 )
        {
          if ( v10 == v12[1] )
          {
            ++v12;
            goto LABEL_12;
          }
          if ( v10 == v12[2] )
          {
            v12 += 2;
            goto LABEL_12;
          }
          if ( v10 == v12[3] )
          {
            v12 += 3;
            goto LABEL_12;
          }
          v12 += 4;
          if ( v17 == v12 )
          {
            v15 = v14 - v12;
            goto LABEL_29;
          }
        }
        goto LABEL_12;
      }
LABEL_29:
      if ( v15 != 2 )
      {
        if ( v15 != 3 )
        {
          if ( v15 != 1 || v10 != *v12 )
            goto LABEL_19;
          goto LABEL_12;
        }
        if ( v10 == *v12 )
        {
LABEL_12:
          if ( v14 == v12 )
            goto LABEL_19;
          goto LABEL_13;
        }
        ++v12;
      }
      if ( v10 != *v12 && v10 != *++v12 )
        goto LABEL_19;
      goto LABEL_12;
    }
    if ( !(_DWORD)v33 )
      goto LABEL_19;
    v18 = (v33 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
    v15 = *(_QWORD *)(v31 + 8LL * v18);
    if ( v10 != v15 )
      break;
LABEL_13:
    if ( v5 == v8 )
      goto LABEL_20;
LABEL_14:
    v9 = v32;
  }
  v19 = 1;
  while ( v15 != -4096 )
  {
    v18 = (v33 - 1) & (v19 + v18);
    v15 = *(_QWORD *)(v31 + 8LL * v18);
    if ( v10 == v15 )
      goto LABEL_13;
    ++v19;
  }
LABEL_19:
  v29 = v8;
  v15 = (__int64)&v30;
  v20 = sub_F61C40(v10, (__int64)&v30, v6, a2);
  v8 = v29;
  v7 |= v20;
  if ( v5 != v29 )
    goto LABEL_14;
LABEL_20:
  for ( i = (unsigned int)v35; (_DWORD)v35; i = (unsigned int)v35 )
  {
    v22 = *(_QWORD *)&v34[8 * i - 8];
    if ( (_DWORD)v33 )
    {
      v23 = (v33 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v24 = (__int64 *)(v31 + 8LL * v23);
      v25 = *v24;
      if ( v22 == *v24 )
      {
LABEL_23:
        *v24 = -8192;
        LODWORD(v32) = v32 - 1;
        ++HIDWORD(v32);
      }
      else
      {
        v27 = 1;
        while ( v25 != -4096 )
        {
          v28 = v27 + 1;
          v23 = (v33 - 1) & (v27 + v23);
          v24 = (__int64 *)(v31 + 8LL * v23);
          v25 = *v24;
          if ( v22 == *v24 )
            goto LABEL_23;
          v27 = v28;
        }
      }
    }
    v15 = (__int64)&v30;
    LODWORD(v35) = v35 - 1;
    v7 |= sub_F61C40(v22, (__int64)&v30, v6, a2);
  }
  if ( v34 != v36 )
    _libc_free(v34, v15);
LABEL_27:
  sub_C7D6A0(v31, 8LL * (unsigned int)v33, 8);
  return v7;
}
