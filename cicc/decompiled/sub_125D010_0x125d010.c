// Function: sub_125D010
// Address: 0x125d010
//
__int64 __fastcall sub_125D010(_QWORD *a1, _DWORD *a2, const char *a3, _QWORD *a4)
{
  _QWORD *v6; // rbx
  size_t v7; // r12
  size_t v8; // rdx
  __int64 result; // rax
  size_t v10; // r12
  _BYTE *v11; // rdi
  size_t v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 v15; // rsi
  size_t v16; // rdx
  _QWORD *v18; // [rsp+10h] [rbp-50h] BYREF
  size_t n; // [rsp+18h] [rbp-48h]
  _QWORD src[8]; // [rsp+20h] [rbp-40h] BYREF

  v6 = (_QWORD *)(*a1 + 32LL * (unsigned int)*a2);
  v7 = strlen(a3);
  if ( (unsigned int)sub_2241B30(v6, 0, v7, a3) )
    return 0;
  v8 = v6[1];
  if ( v8 == v7 )
  {
    v15 = (unsigned int)(*a2 + 1);
    *a2 = v15;
    sub_2240AE0(a4, *a1 + 32 * v15);
    return 1;
  }
  if ( v8 <= v7 )
    sub_222CF80("basic_string::at: __n (which is %zu) >= this->size() (which is %zu)", v7);
  result = 0;
  if ( *(_BYTE *)(*v6 + v7) == 61 )
  {
    v10 = v7 + 1;
    if ( v10 > v8 )
      sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::substr");
    v18 = src;
    sub_125C5B0((__int64 *)&v18, (_BYTE *)(v10 + *v6), *v6 + v8);
    v11 = (_BYTE *)*a4;
    if ( v18 == src )
    {
      v16 = n;
      if ( n )
      {
        if ( n == 1 )
          *v11 = src[0];
        else
          memcpy(v11, src, n);
        v16 = n;
        v11 = (_BYTE *)*a4;
      }
      a4[1] = v16;
      v11[v16] = 0;
      v11 = v18;
      goto LABEL_10;
    }
    v12 = n;
    v13 = src[0];
    if ( v11 == (_BYTE *)(a4 + 2) )
    {
      *a4 = v18;
      a4[1] = v12;
      a4[2] = v13;
    }
    else
    {
      v14 = a4[2];
      *a4 = v18;
      a4[1] = v12;
      a4[2] = v13;
      if ( v11 )
      {
        v18 = v11;
        src[0] = v14;
LABEL_10:
        n = 0;
        *v11 = 0;
        if ( v18 != src )
          j_j___libc_free_0(v18, src[0] + 1LL);
        return 1;
      }
    }
    v18 = src;
    v11 = src;
    goto LABEL_10;
  }
  return result;
}
