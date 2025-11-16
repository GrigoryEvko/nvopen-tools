// Function: sub_1AD5550
// Address: 0x1ad5550
//
void __fastcall sub_1AD5550(__int64 a1, __int64 a2)
{
  _BYTE *v2; // r15
  __int64 v3; // r13
  __int64 *v6; // rsi
  __int64 v7; // rdx
  _BYTE *v8; // rdi
  size_t v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rsi
  size_t v12; // rdx
  _QWORD *v13; // [rsp+0h] [rbp-50h] BYREF
  size_t n; // [rsp+8h] [rbp-48h]
  _QWORD src[8]; // [rsp+10h] [rbp-40h] BYREF

  v2 = (_BYTE *)(a1 + 16);
  v3 = a1 + 32;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  v6 = *(__int64 **)(a2 + 16);
  *(_BYTE *)(a1 + 16) = 0;
  v7 = *v6;
  v13 = src;
  sub_1AD2E00((__int64 *)&v13, (_BYTE *)v6 + 16, (__int64)v6 + v7 + 16);
  v8 = *(_BYTE **)a1;
  if ( v13 == src )
  {
    v12 = n;
    if ( n )
    {
      if ( n == 1 )
        *v8 = src[0];
      else
        memcpy(v8, src, n);
      v12 = n;
      v8 = *(_BYTE **)a1;
    }
    *(_QWORD *)(a1 + 8) = v12;
    v8[v12] = 0;
    v8 = v13;
  }
  else
  {
    v9 = n;
    v10 = src[0];
    if ( v2 == v8 )
    {
      *(_QWORD *)a1 = v13;
      *(_QWORD *)(a1 + 8) = v9;
      *(_QWORD *)(a1 + 16) = v10;
    }
    else
    {
      v11 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)a1 = v13;
      *(_QWORD *)(a1 + 8) = v9;
      *(_QWORD *)(a1 + 16) = v10;
      if ( v8 )
      {
        v13 = v8;
        src[0] = v11;
        goto LABEL_5;
      }
    }
    v13 = src;
    v8 = src;
  }
LABEL_5:
  n = 0;
  *v8 = 0;
  if ( v13 != src )
    j_j___libc_free_0(v13, src[0] + 1LL);
  sub_1AD2EB0(v3, *(char **)(a1 + 40), *(char **)a2, (char *)(*(_QWORD *)a2 + 24LL * *(_QWORD *)(a2 + 8)));
}
