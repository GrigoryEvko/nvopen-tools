// Function: sub_36F0E80
// Address: 0x36f0e80
//
__int64 __fastcall sub_36F0E80(__int64 a1, __int64 *a2)
{
  _QWORD *v4; // rax
  _BYTE *v5; // r12
  __int64 v6; // r8
  __int64 v7; // r9
  _BYTE *v8; // rax
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rcx
  int v11; // r15d
  _BYTE *v12; // rsi
  _BYTE *v13; // rdi
  __int64 v14; // rdx
  unsigned __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rsi
  _BYTE *v19; // rdx
  char v20; // cl
  __int64 v21; // rax
  _BYTE *v22; // rdx
  __int64 v23; // rdi
  char v24; // si
  _BYTE v25[16]; // [rsp+0h] [rbp-D0h] BYREF
  __int64 v26; // [rsp+10h] [rbp-C0h]
  unsigned int v27; // [rsp+18h] [rbp-B8h]
  _BYTE *v28; // [rsp+50h] [rbp-80h]
  __int64 v29; // [rsp+58h] [rbp-78h]
  _BYTE src[112]; // [rsp+60h] [rbp-70h] BYREF

  *(_QWORD *)(a1 + 1136) = a2[2];
  v4 = (_QWORD *)sub_B2BE50(*a2);
  sub_36F0D50((__int64)v25, v4);
  if ( (*(_BYTE *)(a1 + 984) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 992), 8LL * *(unsigned int *)(a1 + 1000), 4);
  v5 = (_BYTE *)(a1 + 992);
  *(_QWORD *)(a1 + 984) = 1;
  do
  {
    if ( v5 )
      *v5 = -1;
    v5 += 8;
  }
  while ( v5 != (_BYTE *)(a1 + 1056) );
  sub_36F0410(a1 + 976, (__int64)v25);
  v8 = v28;
  if ( v28 == src )
  {
    v9 = (unsigned int)v29;
    v10 = *(unsigned int *)(a1 + 1064);
    v11 = v29;
    if ( (unsigned int)v29 <= v10 )
    {
      v13 = src;
      if ( (_DWORD)v29 )
      {
        v17 = *(_QWORD *)(a1 + 1056);
        v18 = v17 + 8LL * (unsigned int)v29;
        v19 = src;
        do
        {
          v20 = *v19;
          v17 += 8;
          v19 += 8;
          *(_BYTE *)(v17 - 8) = v20;
          *(_DWORD *)(v17 - 4) = *((_DWORD *)v19 - 1);
        }
        while ( v17 != v18 );
        goto LABEL_13;
      }
    }
    else
    {
      if ( (unsigned int)v29 <= (unsigned __int64)*(unsigned int *)(a1 + 1068) )
      {
        v12 = src;
        v13 = src;
        if ( *(_DWORD *)(a1 + 1064) )
        {
          v21 = *(_QWORD *)(a1 + 1056);
          v10 *= 8LL;
          v22 = src;
          v23 = v21 + v10;
          do
          {
            v24 = *v22;
            v21 += 8;
            v22 += 8;
            *(_BYTE *)(v21 - 8) = v24;
            *(_DWORD *)(v21 - 4) = *((_DWORD *)v22 - 1);
          }
          while ( v21 != v23 );
          v13 = v28;
          v9 = (unsigned int)v29;
          v12 = &v28[v10];
        }
        v14 = 8 * v9;
        if ( v12 == &v13[v14] )
          goto LABEL_14;
        goto LABEL_12;
      }
      *(_DWORD *)(a1 + 1064) = 0;
      sub_C8D5F0((__int64)v5, (const void *)(a1 + 1072), v9, 8u, v6, v7);
      v13 = v28;
      v10 = 0;
      v14 = 8LL * (unsigned int)v29;
      v12 = v28;
      if ( v28 != &v28[v14] )
      {
LABEL_12:
        memcpy((void *)(v10 + *(_QWORD *)(a1 + 1056)), v12, v14 - v10);
LABEL_13:
        v13 = v28;
      }
    }
LABEL_14:
    *(_DWORD *)(a1 + 1064) = v11;
    if ( v13 != src )
      _libc_free((unsigned __int64)v13);
    goto LABEL_16;
  }
  v16 = *(_QWORD *)(a1 + 1056);
  if ( v16 != a1 + 1072 )
  {
    _libc_free(v16);
    v8 = v28;
  }
  *(_QWORD *)(a1 + 1056) = v8;
  *(_QWORD *)(a1 + 1064) = v29;
LABEL_16:
  if ( (v25[8] & 1) == 0 )
    sub_C7D6A0(v26, 8LL * v27, 4);
  return sub_34318E0(a1, a2);
}
