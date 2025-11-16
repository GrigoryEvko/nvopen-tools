// Function: sub_371E590
// Address: 0x371e590
//
void __fastcall sub_371E590(_QWORD *p_src, __int64 a2)
{
  _QWORD *v2; // r15
  char *v4; // r14
  size_t *v5; // rbx
  _BYTE *v6; // rsi
  size_t v7; // rdx
  _BYTE *v8; // r8
  __int64 v9; // rbx
  _BYTE *v10; // rax
  _QWORD *v11; // r14
  bool v12; // zf
  _BYTE *v13; // [rsp+8h] [rbp-78h]
  _QWORD v14[2]; // [rsp+20h] [rbp-60h] BYREF
  void *src; // [rsp+30h] [rbp-50h] BYREF
  _BYTE *v16; // [rsp+38h] [rbp-48h]
  _BYTE *v17; // [rsp+40h] [rbp-40h]

  v2 = (_QWORD *)*p_src;
  if ( *p_src )
  {
    while ( 1 )
    {
      while ( !(-1431655765 * (unsigned int)((__int64)(v2[2] - v2[1]) >> 3)) )
      {
LABEL_3:
        v2 = (_QWORD *)*v2;
        if ( !v2 )
          return;
      }
      v17 = 0;
      v4 = (char *)v2[2];
      v16 = 0;
      v5 = (size_t *)v2[1];
      src = 0;
      v6 = (_BYTE *)(v4 - (char *)v5);
      v7 = 0xAAAAAAAAAAAAAAABLL * ((v4 - (char *)v5) >> 3);
      if ( v4 - (char *)v5 < 0 )
        sub_4262D8((__int64)"vector::reserve");
      v8 = 0;
      if ( v7 )
        break;
LABEL_14:
      while ( v4 != (char *)v5 )
      {
        if ( v17 == v8 )
        {
          p_src = &src;
          v6 = v8;
          sub_24454E0((__int64)&src, v8, v5);
          v8 = v16;
        }
        else
        {
          if ( v8 )
          {
            v7 = *v5;
            *(_QWORD *)v8 = *v5;
            v8 = v16;
          }
          v8 += 8;
          v16 = v8;
        }
        v5 += 3;
      }
      v14[0] = src;
      v12 = *(_QWORD *)(a2 + 16) == 0;
      v14[1] = (v8 - (_BYTE *)src) >> 3;
      if ( v12 )
        sub_4263D6(p_src, v6, v7);
      (*(void (__fastcall **)(__int64, _QWORD *))(a2 + 24))(a2, v14);
      p_src = src;
      if ( !src )
        goto LABEL_3;
      j_j___libc_free_0((unsigned __int64)src);
      v2 = (_QWORD *)*v2;
      if ( !v2 )
        return;
    }
    v9 = 0x5555555555555558LL * ((v4 - (char *)v5) >> 3);
    p_src = (_QWORD *)(8 * v7);
    v10 = (_BYTE *)sub_22077B0(8 * v7);
    v11 = src;
    v8 = v10;
    v7 = v16 - (_BYTE *)src;
    if ( v16 - (_BYTE *)src > 0 )
    {
      v8 = memmove(v10, src, v7);
      v6 = (_BYTE *)(v17 - (_BYTE *)v11);
    }
    else
    {
      if ( !src )
      {
LABEL_9:
        v16 = v8;
        v17 = &v8[v9];
        v4 = (char *)v2[2];
        src = v8;
        v5 = (size_t *)v2[1];
        goto LABEL_14;
      }
      v6 = (_BYTE *)(v17 - (_BYTE *)src);
    }
    p_src = v11;
    v13 = v8;
    j_j___libc_free_0((unsigned __int64)v11);
    v8 = v13;
    goto LABEL_9;
  }
}
