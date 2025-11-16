// Function: sub_15637E0
// Address: 0x15637e0
//
__int64 __fastcall sub_15637E0(__int64 *a1, __int64 *a2, __int32 a3, _QWORD *a4)
{
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 *v9; // rdi
  int v10; // edx
  const void *v11; // r11
  signed __int64 v12; // r8
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r12
  unsigned __int64 v18; // r8
  __int64 *v19; // rax
  __int64 *v20; // rcx
  signed __int64 v21; // [rsp+8h] [rbp-E8h]
  const void *v22; // [rsp+10h] [rbp-E0h]
  __int64 v23; // [rsp+20h] [rbp-D0h]
  unsigned int v24; // [rsp+28h] [rbp-C8h]
  __int64 *v25; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v26; // [rsp+38h] [rbp-B8h]
  _BYTE dest[32]; // [rsp+40h] [rbp-B0h] BYREF
  __m128i v28; // [rsp+60h] [rbp-90h] BYREF
  _QWORD *v29; // [rsp+78h] [rbp-78h]

  if ( !sub_1560CB0(a4) )
    return *a1;
  if ( *a1 )
  {
    if ( a3 == -1 )
    {
      v23 = 0;
      v24 = 0;
    }
    else
    {
      v24 = a3 + 1;
      v23 = (unsigned int)(a3 + 1);
    }
    v7 = sub_15601B0(a1);
    v8 = sub_15601A0(a1);
    v9 = (__int64 *)dest;
    v10 = 0;
    v11 = (const void *)v8;
    v25 = (__int64 *)dest;
    v12 = v7 - v8;
    v26 = 0x400000000LL;
    v13 = (v7 - v8) >> 3;
    if ( (unsigned __int64)(v7 - v8) > 0x20 )
    {
      v21 = v7 - v8;
      v22 = (const void *)v8;
      sub_16CD150(&v25, dest, v12 >> 3, 8);
      v10 = v26;
      v12 = v21;
      v11 = v22;
      v9 = &v25[(unsigned int)v26];
    }
    if ( (const void *)v7 != v11 )
    {
      memcpy(v9, v11, v12);
      v10 = v26;
    }
    LODWORD(v14) = v13 + v10;
    LODWORD(v26) = v13 + v10;
    if ( v24 >= (int)v13 + v10 )
    {
      v14 = (unsigned int)v14;
      v18 = v24 + 1;
      if ( v18 >= (unsigned int)v14 )
      {
        if ( v18 > (unsigned int)v14 )
        {
          if ( v18 > HIDWORD(v26) )
          {
            sub_16CD150(&v25, dest, v24 + 1, 8);
            v14 = (unsigned int)v26;
            v18 = v24 + 1;
          }
          v15 = v25;
          v19 = &v25[v14];
          v20 = &v25[v18];
          if ( v19 != v20 )
          {
            do
            {
              if ( v19 )
                *v19 = 0;
              ++v19;
            }
            while ( v20 != v19 );
            v15 = v25;
          }
          LODWORD(v26) = v24 + 1;
          goto LABEL_12;
        }
      }
      else
      {
        LODWORD(v26) = v24 + 1;
      }
    }
    v15 = v25;
LABEL_12:
    sub_1563030(&v28, v15[v23]);
    sub_15625F0(&v28, a4);
    v16 = sub_1560BF0(a2, &v28);
    v25[v23] = v16;
    v17 = sub_155F990(a2, v25, (unsigned int)v26);
    sub_155CC10(v29);
    if ( v25 != (__int64 *)dest )
      _libc_free((unsigned __int64)v25);
    return v17;
  }
  v28.m128i_i32[0] = a3;
  v28.m128i_i64[1] = sub_1560BF0(a2, a4);
  return sub_155FA70(a2, v28.m128i_i32, 1);
}
