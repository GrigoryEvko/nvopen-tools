// Function: sub_147AA20
// Address: 0x147aa20
//
__int64 __fastcall sub_147AA20(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // rsi
  unsigned int v6; // eax
  unsigned int *v7; // r14
  __int64 v8; // r12
  unsigned int v9; // eax
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // rcx
  unsigned __int8 v18; // [rsp+28h] [rbp-F8h]
  unsigned __int8 v19; // [rsp+28h] [rbp-F8h]
  unsigned int v20; // [rsp+3Ch] [rbp-E4h] BYREF
  unsigned __int64 v21; // [rsp+40h] [rbp-E0h] BYREF
  unsigned int v22; // [rsp+48h] [rbp-D8h]
  unsigned __int64 i; // [rsp+50h] [rbp-D0h] BYREF
  unsigned int v24; // [rsp+58h] [rbp-C8h]
  _BYTE *v25; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v26; // [rsp+68h] [rbp-B8h]
  _BYTE v27[176]; // [rsp+70h] [rbp-B0h] BYREF

  result = 0;
  if ( !*(_WORD *)(a2 + 24) )
  {
    v5 = *(_QWORD *)(a2 + 32);
    v6 = *(_DWORD *)(v5 + 32);
    v22 = v6;
    if ( v6 > 0x40 )
    {
      sub_16A4FD0(&v21, v5 + 24);
      v6 = v22;
    }
    else
    {
      v21 = *(_QWORD *)(v5 + 24);
    }
    v24 = v6;
    v7 = (unsigned int *)&unk_428DC40;
    v8 = 4294967294LL;
    if ( v6 > 0x40 )
      goto LABEL_18;
LABEL_5:
    for ( i = v21; ; sub_16A4FD0(&i, &v21) )
    {
      sub_16A7800(&i, v8);
      v9 = v24;
      v24 = 0;
      LODWORD(v26) = v9;
      v25 = (_BYTE *)i;
      v10 = sub_145CF40(a1, (__int64)&v25);
      if ( (unsigned int)v26 > 0x40 && v25 )
        j_j___libc_free_0_0(v25);
      if ( v24 > 0x40 && i )
        j_j___libc_free_0_0(i);
      v25 = v27;
      v26 = 0x2000000000LL;
      sub_16BD3E0(&v25, 7);
      sub_16BD4C0(&v25, v10);
      sub_16BD4C0(&v25, a3);
      sub_16BD4C0(&v25, a4);
      i = 0;
      v11 = sub_16BDDE0(a1 + 816, &v25, &i);
      v12 = v11;
      if ( v11 )
      {
        if ( (*(_BYTE *)(v11 + 26) & 2) != 0 )
        {
          v13 = sub_145CF80(a1, **(_QWORD **)(a2 + 32), v8, 0);
          v20 = 42;
          v14 = sub_1477D10(v13, &v20, a1);
          if ( v14 )
          {
            result = sub_147A340(a1, v20, v12, v14);
            if ( (_BYTE)result )
              break;
          }
        }
      }
      if ( v25 != v27 )
        _libc_free((unsigned __int64)v25);
      if ( jpt_1498F97 == (_UNKNOWN *__ptr32 *)++v7 )
      {
        result = 0;
        goto LABEL_20;
      }
      v8 = *v7;
      v24 = v22;
      if ( v22 <= 0x40 )
        goto LABEL_5;
LABEL_18:
      ;
    }
    if ( v25 != v27 )
    {
      v19 = result;
      _libc_free((unsigned __int64)v25);
      result = v19;
    }
LABEL_20:
    if ( v22 > 0x40 )
    {
      if ( v21 )
      {
        v18 = result;
        j_j___libc_free_0_0(v21);
        return v18;
      }
    }
  }
  return result;
}
