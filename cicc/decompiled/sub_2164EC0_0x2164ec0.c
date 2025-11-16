// Function: sub_2164EC0
// Address: 0x2164ec0
//
__int64 __fastcall sub_2164EC0(__int64 a1, const char *a2, unsigned __int64 a3)
{
  __int64 v3; // r8
  const char *v5; // r9
  size_t v6; // r13
  _QWORD *v7; // rdi
  _QWORD *v8; // rax
  _BYTE *v9; // rdi
  _BYTE *v10; // rax
  size_t v11; // rsi
  __int64 v12; // rcx
  __int64 v13; // rdi
  __int64 v15; // rax
  size_t v16; // rdx
  int v17; // eax
  __int64 v18; // [rsp+8h] [rbp-68h]
  __int64 v19; // [rsp+8h] [rbp-68h]
  __int64 v20; // [rsp+8h] [rbp-68h]
  __int64 v21; // [rsp+18h] [rbp-58h] BYREF
  _QWORD *v22; // [rsp+20h] [rbp-50h] BYREF
  size_t n; // [rsp+28h] [rbp-48h]
  _QWORD dest[8]; // [rsp+30h] [rbp-40h] BYREF

  v3 = a1 + 216;
  if ( !a3 )
  {
    v21 = 5;
    v5 = "sm_52";
    v6 = 5;
    v22 = dest;
    v7 = dest;
LABEL_3:
    v18 = v3;
    memcpy(v7, v5, v6);
    v6 = v21;
    v8 = v22;
    v3 = v18;
    goto LABEL_4;
  }
  v5 = a2;
  if ( !a2 )
  {
    LOBYTE(dest[0]) = 0;
    v9 = *(_BYTE **)(a1 + 216);
    v16 = 0;
    v22 = dest;
LABEL_20:
    *(_QWORD *)(a1 + 224) = v16;
    v9[v16] = 0;
    v10 = v22;
    goto LABEL_8;
  }
  v21 = a3;
  v6 = a3;
  v22 = dest;
  if ( a3 > 0xF )
  {
    v15 = sub_22409D0(&v22, &v21, 0);
    v5 = a2;
    v3 = a1 + 216;
    v22 = (_QWORD *)v15;
    v7 = (_QWORD *)v15;
    dest[0] = v21;
    goto LABEL_3;
  }
  if ( a3 != 1 )
  {
    v7 = dest;
    goto LABEL_3;
  }
  LOBYTE(dest[0]) = *a2;
  v8 = dest;
LABEL_4:
  n = v6;
  *((_BYTE *)v8 + v6) = 0;
  v9 = *(_BYTE **)(a1 + 216);
  v10 = v9;
  if ( v22 == dest )
  {
    v16 = n;
    if ( n )
    {
      if ( n == 1 )
      {
        *v9 = dest[0];
        v16 = n;
      }
      else
      {
        v20 = v3;
        memcpy(v9, dest, n);
        v16 = n;
        v3 = v20;
      }
      v9 = *(_BYTE **)(a1 + 216);
    }
    goto LABEL_20;
  }
  v11 = n;
  v12 = dest[0];
  if ( v9 == (_BYTE *)(a1 + 232) )
  {
    *(_QWORD *)(a1 + 216) = v22;
    *(_QWORD *)(a1 + 224) = v11;
    *(_QWORD *)(a1 + 232) = v12;
  }
  else
  {
    v13 = *(_QWORD *)(a1 + 232);
    *(_QWORD *)(a1 + 216) = v22;
    *(_QWORD *)(a1 + 224) = v11;
    *(_QWORD *)(a1 + 232) = v12;
    if ( v10 )
    {
      v22 = v10;
      dest[0] = v13;
      goto LABEL_8;
    }
  }
  v22 = dest;
  v10 = dest;
LABEL_8:
  n = 0;
  *v10 = 0;
  if ( v22 != dest )
  {
    v19 = v3;
    j_j___libc_free_0(v22, dest[0] + 1LL);
    v3 = v19;
  }
  if ( !sub_22416F0(v3, "sm_", 0, 3) )
  {
    v17 = strtol((const char *)(*(_QWORD *)(a1 + 216) + 3LL), 0, 10);
    if ( !v17 )
      v17 = 52;
    *(_DWORD *)(a1 + 252) = v17;
  }
  *(_QWORD *)(a1 + 82312) = -1;
  *(_QWORD *)(a1 + 82304) = 0xFFFFFFFF00000000LL;
  *(_DWORD *)(a1 + 82320) = 0;
  sub_21642F0(a1);
  if ( *(_DWORD *)(a1 + 82320) )
    *(_DWORD *)(a1 + 82304) = 32;
  else
    *(_DWORD *)(a1 + 82304) = *(_BYTE *)(*(_QWORD *)(a1 + 256) + 936LL) == 0 ? 32 : 64;
  if ( !*(_DWORD *)(a1 + 248) )
    *(_DWORD *)(a1 + 248) = 90;
  return a1;
}
