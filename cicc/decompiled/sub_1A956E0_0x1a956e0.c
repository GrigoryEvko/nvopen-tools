// Function: sub_1A956E0
// Address: 0x1a956e0
//
__int64 *__fastcall sub_1A956E0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, _BYTE *a5, size_t a6)
{
  char v7; // al
  __int64 v8; // rdx
  _BYTE *v10; // rdi
  size_t v11; // rax
  _QWORD **v12; // rcx
  __int64 v13; // rax
  size_t n; // [rsp+0h] [rbp-80h]
  _BYTE *src; // [rsp+8h] [rbp-78h]
  _QWORD v16[2]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD v17[2]; // [rsp+20h] [rbp-60h] BYREF
  _QWORD *v18; // [rsp+30h] [rbp-50h] BYREF
  __int16 v19; // [rsp+40h] [rbp-40h]
  _QWORD v20[2]; // [rsp+50h] [rbp-30h] BYREF
  __int16 v21; // [rsp+60h] [rbp-20h]

  v16[0] = a3;
  v16[1] = a4;
  if ( (*(_BYTE *)(a2 + 23) & 0x20) == 0 )
  {
    v10 = a1 + 2;
    if ( !a5 )
    {
      *a1 = (__int64)v10;
      a1[1] = 0;
      *((_BYTE *)a1 + 16) = 0;
      return a1;
    }
    *a1 = (__int64)v10;
    v11 = a6;
    v20[0] = a6;
    if ( a6 > 0xF )
    {
      n = a6;
      src = a5;
      v13 = sub_22409D0(a1, v20, 0);
      a5 = src;
      a6 = n;
      *a1 = v13;
      v10 = (_BYTE *)v13;
      a1[2] = v20[0];
    }
    else
    {
      if ( a6 == 1 )
      {
        *((_BYTE *)a1 + 16) = *a5;
LABEL_9:
        a1[1] = v11;
        v10[v11] = 0;
        return a1;
      }
      if ( !a6 )
        goto LABEL_9;
    }
    memcpy(v10, a5, a6);
    v11 = v20[0];
    v10 = (_BYTE *)*a1;
    goto LABEL_9;
  }
  v19 = 261;
  v18 = v16;
  v17[0] = sub_1649960(a2);
  v7 = v19;
  v17[1] = v8;
  if ( (_BYTE)v19 )
  {
    if ( (_BYTE)v19 == 1 )
    {
      v20[0] = v17;
      v21 = 261;
    }
    else
    {
      v12 = (_QWORD **)v18;
      if ( HIBYTE(v19) != 1 )
      {
        v12 = &v18;
        v7 = 2;
      }
      v20[1] = v12;
      v20[0] = v17;
      LOBYTE(v21) = 5;
      HIBYTE(v21) = v7;
    }
  }
  else
  {
    v21 = 256;
  }
  sub_16E2FC0(a1, (__int64)v20);
  return a1;
}
