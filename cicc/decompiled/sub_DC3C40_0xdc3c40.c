// Function: sub_DC3C40
// Address: 0xdc3c40
//
__int64 __fastcall sub_DC3C40(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // rsi
  unsigned int v6; // eax
  __int128 *v7; // rbx
  unsigned __int64 v8; // r12
  unsigned int v9; // eax
  _QWORD *v10; // rax
  unsigned int *v11; // rsi
  _QWORD *v12; // rax
  _QWORD *v13; // rax
  _BYTE *v14; // rcx
  unsigned __int64 v15; // [rsp+0h] [rbp-130h]
  _QWORD *v16; // [rsp+8h] [rbp-128h]
  _QWORD *v17; // [rsp+8h] [rbp-128h]
  _BYTE *v18; // [rsp+8h] [rbp-128h]
  unsigned __int8 v22; // [rsp+38h] [rbp-F8h]
  unsigned __int8 v23; // [rsp+38h] [rbp-F8h]
  unsigned int v24; // [rsp+4Ch] [rbp-E4h] BYREF
  __int64 v25; // [rsp+50h] [rbp-E0h] BYREF
  unsigned int v26; // [rsp+58h] [rbp-D8h]
  int *i; // [rsp+60h] [rbp-D0h] BYREF
  unsigned int v28; // [rsp+68h] [rbp-C8h]
  int *v29; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v30; // [rsp+78h] [rbp-B8h]
  int v31; // [rsp+80h] [rbp-B0h] BYREF
  _QWORD *v32; // [rsp+84h] [rbp-ACh]
  __int64 v33; // [rsp+8Ch] [rbp-A4h]
  __int64 v34; // [rsp+94h] [rbp-9Ch]

  result = 0;
  if ( !*(_WORD *)(a2 + 24) )
  {
    v5 = *(_QWORD *)(a2 + 32);
    v6 = *(_DWORD *)(v5 + 32);
    v26 = v6;
    if ( v6 > 0x40 )
    {
      sub_C43780((__int64)&v25, (const void **)(v5 + 24));
      v6 = v26;
    }
    else
    {
      v25 = *(_QWORD *)(v5 + 24);
    }
    v28 = v6;
    v7 = (__int128 *)&unk_3F74E90;
    v8 = 4294967294LL;
    if ( v6 > 0x40 )
      goto LABEL_18;
LABEL_5:
    for ( i = (int *)v25; ; sub_C43780((__int64)&i, (const void **)&v25) )
    {
      sub_C46F20((__int64)&i, v8);
      v9 = v28;
      v28 = 0;
      LODWORD(v30) = v9;
      v29 = i;
      v10 = sub_DA26C0(a1, (__int64)&v29);
      if ( (unsigned int)v30 > 0x40 && v29 )
      {
        v16 = v10;
        j_j___libc_free_0_0(v29);
        v10 = v16;
      }
      if ( v28 > 0x40 && i )
      {
        v17 = v10;
        j_j___libc_free_0_0(i);
        v10 = v17;
      }
      v32 = v10;
      v11 = (unsigned int *)&v29;
      v31 = 8;
      v33 = a3;
      v29 = &v31;
      v34 = a4;
      v30 = 0x2000000007LL;
      i = 0;
      v12 = sub_C65B40((__int64)(a1 + 129), (__int64)&v29, (__int64 *)&i, (__int64)off_49DEA80);
      if ( v12 )
      {
        if ( (*((_BYTE *)v12 + 28) & 2) != 0 )
        {
          v18 = v12;
          v13 = sub_DA2C50((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), v8, 0);
          v11 = &v24;
          v24 = 42;
          v14 = sub_DBEE70((__int64)v13, &v24, a1);
          if ( v14 )
          {
            v15 = v24 | v15 & 0xFFFFFF0000000000LL;
            v11 = (unsigned int *)v15;
            result = sub_DC3A60((__int64)a1, v15, v18, v14);
            if ( (_BYTE)result )
              break;
          }
        }
      }
      if ( v29 != &v31 )
        _libc_free(v29, v11);
      v7 = (__int128 *)((char *)v7 + 4);
      if ( v7 == &xmmword_3F74EA0 )
      {
        result = 0;
        goto LABEL_20;
      }
      v8 = *(unsigned int *)v7;
      v28 = v26;
      if ( v26 <= 0x40 )
        goto LABEL_5;
LABEL_18:
      ;
    }
    if ( v29 != &v31 )
    {
      v23 = result;
      _libc_free(v29, v15);
      result = v23;
    }
LABEL_20:
    if ( v26 > 0x40 )
    {
      if ( v25 )
      {
        v22 = result;
        j_j___libc_free_0_0(v25);
        return v22;
      }
    }
  }
  return result;
}
