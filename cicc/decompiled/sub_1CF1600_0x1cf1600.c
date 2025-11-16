// Function: sub_1CF1600
// Address: 0x1cf1600
//
unsigned __int64 __fastcall sub_1CF1600(_QWORD *p_src, __int64 a2)
{
  _QWORD *v2; // r15
  unsigned __int64 result; // rax
  char *v5; // r14
  size_t *v6; // rbx
  _BYTE *v7; // rsi
  size_t v8; // rdx
  _BYTE *v9; // r8
  __int64 v10; // rbx
  _BYTE *v11; // rax
  _QWORD *v12; // r14
  bool v13; // zf
  _BYTE *v14; // [rsp+8h] [rbp-78h]
  _QWORD v15[2]; // [rsp+20h] [rbp-60h] BYREF
  void *src; // [rsp+30h] [rbp-50h] BYREF
  _BYTE *v17; // [rsp+38h] [rbp-48h]
  _BYTE *v18; // [rsp+40h] [rbp-40h]

  v2 = (_QWORD *)*p_src;
  result = (unsigned __int64)&src;
  if ( *p_src )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        result = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v2[2] - v2[1]) >> 3);
        if ( (_DWORD)result )
          break;
LABEL_3:
        v2 = (_QWORD *)*v2;
        if ( !v2 )
          return result;
      }
      v18 = 0;
      v5 = (char *)v2[2];
      v17 = 0;
      v6 = (size_t *)v2[1];
      src = 0;
      v7 = (_BYTE *)(v5 - (char *)v6);
      v8 = 0xAAAAAAAAAAAAAAABLL * ((v5 - (char *)v6) >> 3);
      if ( v5 - (char *)v6 < 0 )
        sub_4262D8((__int64)"vector::reserve");
      v9 = 0;
      if ( v8 )
        break;
LABEL_14:
      while ( v5 != (char *)v6 )
      {
        if ( v18 == v9 )
        {
          p_src = &src;
          v7 = v9;
          sub_170B610((__int64)&src, v9, v6);
          v9 = v17;
        }
        else
        {
          if ( v9 )
          {
            v8 = *v6;
            *(_QWORD *)v9 = *v6;
            v9 = v17;
          }
          v9 += 8;
          v17 = v9;
        }
        v6 += 3;
      }
      v15[0] = src;
      v13 = *(_QWORD *)(a2 + 16) == 0;
      v15[1] = (v9 - (_BYTE *)src) >> 3;
      if ( v13 )
        sub_4263D6(p_src, v7, v8);
      result = (*(__int64 (__fastcall **)(__int64, _QWORD *))(a2 + 24))(a2, v15);
      p_src = src;
      if ( !src )
        goto LABEL_3;
      result = j_j___libc_free_0(src, v18 - (_BYTE *)src);
      v2 = (_QWORD *)*v2;
      if ( !v2 )
        return result;
    }
    v10 = 0x5555555555555558LL * ((v5 - (char *)v6) >> 3);
    p_src = (_QWORD *)(8 * v8);
    v11 = (_BYTE *)sub_22077B0(8 * v8);
    v12 = src;
    v9 = v11;
    v8 = v17 - (_BYTE *)src;
    if ( v17 - (_BYTE *)src > 0 )
    {
      v9 = memmove(v11, src, v8);
      v7 = (_BYTE *)(v18 - (_BYTE *)v12);
    }
    else
    {
      if ( !src )
      {
LABEL_9:
        v17 = v9;
        v18 = &v9[v10];
        v5 = (char *)v2[2];
        src = v9;
        v6 = (size_t *)v2[1];
        goto LABEL_14;
      }
      v7 = (_BYTE *)(v18 - (_BYTE *)src);
    }
    p_src = v12;
    v14 = v9;
    j_j___libc_free_0(v12, v7);
    v9 = v14;
    goto LABEL_9;
  }
  return result;
}
