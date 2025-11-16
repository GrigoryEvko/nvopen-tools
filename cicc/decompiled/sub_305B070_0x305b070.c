// Function: sub_305B070
// Address: 0x305b070
//
__int64 __fastcall sub_305B070(__int64 a1, char *a2, __int64 a3, _BYTE *a4, __int64 a5)
{
  _QWORD *v8; // rdi
  __int64 *v9; // r8
  __int64 v10; // rdx
  size_t v11; // rcx
  __int64 v12; // rsi
  char *v13; // rdx
  char v14; // al
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // r9
  _DWORD *v18; // rcx
  size_t v19; // r8
  int v21; // eax
  size_t v22; // rdx
  size_t v23; // [rsp+0h] [rbp-80h]
  __int64 *v24; // [rsp+8h] [rbp-78h]
  __int64 v25; // [rsp+8h] [rbp-78h]
  _DWORD *v26[2]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD v27[2]; // [rsp+20h] [rbp-60h] BYREF
  _QWORD *v28; // [rsp+30h] [rbp-50h] BYREF
  size_t n; // [rsp+38h] [rbp-48h]
  _QWORD src[8]; // [rsp+40h] [rbp-40h] BYREF

  if ( !a3 )
  {
    a3 = 5;
    a2 = "sm_52";
  }
  v28 = src;
  sub_3059660((__int64 *)&v28, a2, (__int64)&a2[a3]);
  v8 = *(_QWORD **)(a1 + 304);
  v9 = (__int64 *)(a1 + 304);
  if ( v28 == src )
  {
    v22 = n;
    if ( n )
    {
      if ( n == 1 )
      {
        *(_BYTE *)v8 = src[0];
        v22 = n;
      }
      else
      {
        memcpy(v8, src, n);
        v22 = n;
        v9 = (__int64 *)(a1 + 304);
      }
      v8 = *(_QWORD **)(a1 + 304);
    }
    *(_QWORD *)(a1 + 312) = v22;
    *((_BYTE *)v8 + v22) = 0;
    v8 = v28;
  }
  else
  {
    v10 = src[0];
    v11 = n;
    if ( v8 == (_QWORD *)(a1 + 320) )
    {
      *(_QWORD *)(a1 + 304) = v28;
      *(_QWORD *)(a1 + 312) = v11;
      *(_QWORD *)(a1 + 320) = v10;
    }
    else
    {
      v12 = *(_QWORD *)(a1 + 320);
      *(_QWORD *)(a1 + 304) = v28;
      *(_QWORD *)(a1 + 312) = v11;
      *(_QWORD *)(a1 + 320) = v10;
      if ( v8 )
      {
        v28 = v8;
        src[0] = v12;
        goto LABEL_7;
      }
    }
    v28 = src;
    v8 = src;
  }
LABEL_7:
  n = 0;
  *(_BYTE *)v8 = 0;
  if ( v28 != src )
  {
    v24 = v9;
    j_j___libc_free_0((unsigned __int64)v28);
    v9 = v24;
  }
  v13 = (char *)(*(_QWORD *)(a1 + 304) + *(_QWORD *)(a1 + 312) - 1LL);
  v14 = *v13;
  if ( *v13 == 97 )
  {
    *(_BYTE *)(a1 + 348) = 1;
    v14 = *v13;
  }
  if ( v14 == 102 )
    *(_BYTE *)(a1 + 349) = 1;
  if ( !sub_22416F0(v9, "sm_", 0, 3u) )
  {
    v21 = 10 * strtol((const char *)(*(_QWORD *)(a1 + 304) + 3LL), 0, 10);
    if ( !v21 )
      v21 = 520;
    *(_DWORD *)(a1 + 340) = v21;
  }
  v15 = *(_QWORD *)(a1 + 312);
  *(_QWORD *)(a1 + 352) = -1;
  *(_QWORD *)(a1 + 360) = 0xFFFFFFFFLL;
  *(_BYTE *)(a1 + 368) = 0;
  v28 = src;
  if ( v15 )
    sub_30597C0((__int64 *)&v28, *(_BYTE **)(a1 + 304), *(_QWORD *)(a1 + 304) + v15);
  else
    sub_3059660((__int64 *)&v28, "sm_30", (__int64)"");
  v16 = *(_QWORD *)(a1 + 312);
  v26[0] = v27;
  v23 = n;
  v25 = (__int64)v28;
  if ( v16 )
  {
    sub_30597C0((__int64 *)v26, *(_BYTE **)(a1 + 304), *(_QWORD *)(a1 + 304) + v16);
    v19 = v23;
    v18 = (_DWORD *)v25;
  }
  else
  {
    sub_3059660((__int64 *)v26, "sm_30", (__int64)"");
    v18 = (_DWORD *)v25;
    v19 = v23;
  }
  sub_3059CE0(a1, v26[0], (size_t)v26[1], v18, v19, v17, a4, a5);
  if ( (_QWORD *)v26[0] != v27 )
    j_j___libc_free_0((unsigned __int64)v26[0]);
  if ( v28 != src )
    j_j___libc_free_0((unsigned __int64)v28);
  *(_DWORD *)(a1 + 344) = *(_DWORD *)(a1 + 340) / 0xAu;
  if ( !*(_DWORD *)(a1 + 336) )
    *(_DWORD *)(a1 + 336) = 90;
  return a1;
}
