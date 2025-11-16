// Function: sub_16A21B0
// Address: 0x16a21b0
//
__int64 __fastcall sub_16A21B0(__int64 a1, char a2, double a3, double a4, double a5)
{
  __int16 *v5; // rax
  __int16 *v6; // rbx
  __int64 v7; // rax
  __int16 *v8; // rdx
  __int16 **v9; // r8
  __int64 v10; // rax
  __int16 *v11; // r15
  __int16 **v12; // r14
  __int64 result; // rax
  __int64 v14; // rdx
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rbx
  __int64 v21; // rcx
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rax
  __int16 *v25; // [rsp+8h] [rbp-88h]
  __int64 v26; // [rsp+10h] [rbp-80h]
  __int64 v27; // [rsp+18h] [rbp-78h]
  __int16 **v28; // [rsp+18h] [rbp-78h]
  _QWORD *v29; // [rsp+20h] [rbp-70h]
  __int64 v30; // [rsp+20h] [rbp-70h]
  __int64 v31; // [rsp+20h] [rbp-70h]
  __int64 v32; // [rsp+20h] [rbp-70h]
  __int16 **v33; // [rsp+20h] [rbp-70h]
  __int64 v35; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v36; // [rsp+38h] [rbp-58h]
  __int16 *v37; // [rsp+48h] [rbp-48h] BYREF
  __int64 v38; // [rsp+50h] [rbp-40h]

  v36 = 64;
  v35 = 0x7FEFFFFFFFFFFFFFLL;
  v5 = (__int16 *)sub_16982C0();
  v6 = v5;
  if ( v5 == word_42AE9D0 )
    sub_169D060(&v37, (__int64)v5, &v35);
  else
    sub_169D050((__int64)&v37, word_42AE9D0, &v35);
  v7 = *(_QWORD *)(a1 + 8);
  v8 = v37;
  v9 = (__int16 **)(v7 + 8);
  if ( *(__int16 **)(v7 + 8) == v6 )
  {
    if ( v6 == v37 )
    {
      if ( v9 == &v37 )
        goto LABEL_24;
      v21 = *(_QWORD *)(v7 + 16);
      if ( v21 )
      {
        v22 = 32LL * *(_QWORD *)(v21 - 8);
        v23 = v21 + v22;
        while ( v21 != v23 )
        {
          v24 = v23 - 32;
          v25 = v8;
          v26 = v21;
          v28 = v9;
          v32 = v24;
          if ( *(__int16 **)(v24 + 8) == v8 )
          {
            sub_169DEB0((__int64 *)(v24 + 16));
            v8 = v25;
            v21 = v26;
            v9 = v28;
            v23 = v32;
          }
          else
          {
            sub_1698460(v24 + 8);
            v23 = v32;
            v9 = v28;
            v21 = v26;
            v8 = v25;
          }
        }
        v33 = v9;
        j_j_j___libc_free_0_0(v21 - 8);
        v9 = v33;
      }
      sub_169C7E0(v9, &v37);
      goto LABEL_6;
    }
    if ( v9 == &v37 )
      goto LABEL_7;
  }
  else
  {
    if ( v6 != v37 )
    {
      sub_16983E0(v7 + 8, (__int64)&v37);
      goto LABEL_6;
    }
    if ( v9 == &v37 )
      goto LABEL_24;
  }
  v29 = (_QWORD *)(v7 + 8);
  sub_127D120((_QWORD *)(v7 + 8));
  if ( v6 != v37 )
  {
    sub_1698450((__int64)v29, (__int64)&v37);
    if ( v37 != v6 )
      goto LABEL_7;
    goto LABEL_24;
  }
  sub_169C7E0(v29, &v37);
LABEL_6:
  if ( v37 != v6 )
  {
LABEL_7:
    sub_1698460((__int64)&v37);
    goto LABEL_8;
  }
LABEL_24:
  v14 = v38;
  if ( v38 )
  {
    v15 = 32LL * *(_QWORD *)(v38 - 8);
    v16 = v38 + v15;
    if ( v38 != v38 + v15 )
    {
      do
      {
        v17 = v16 - 32;
        v27 = v14;
        v30 = v17;
        if ( v6 == *(__int16 **)(v17 + 8) )
        {
          sub_169DEB0((__int64 *)(v17 + 16));
          v14 = v27;
          v16 = v30;
        }
        else
        {
          sub_1698460(v17 + 8);
          v16 = v30;
          v14 = v27;
        }
      }
      while ( v14 != v16 );
    }
    j_j_j___libc_free_0_0(v14 - 8);
  }
LABEL_8:
  if ( v36 > 0x40 && v35 )
    j_j___libc_free_0_0(v35);
  v36 = 64;
  v35 = 0x7C8FFFFFFFFFFFFELL;
  if ( v6 == word_42AE9D0 )
    sub_169D060(&v37, (__int64)v6, &v35);
  else
    sub_169D050((__int64)&v37, word_42AE9D0, &v35);
  v10 = *(_QWORD *)(a1 + 8);
  v11 = v37;
  v12 = (__int16 **)(v10 + 40);
  if ( *(__int16 **)(v10 + 40) == v6 )
  {
    if ( v6 == v37 )
    {
      if ( v12 == &v37 )
        goto LABEL_16;
      v18 = *(_QWORD *)(v10 + 48);
      if ( v18 )
      {
        v19 = 32LL * *(_QWORD *)(v18 - 8);
        v20 = v18 + v19;
        if ( v18 != v18 + v19 )
        {
          do
          {
            v20 -= 32;
            v31 = v18;
            if ( v11 == *(__int16 **)(v20 + 8) )
              sub_169DEB0((__int64 *)(v20 + 16));
            else
              sub_1698460(v20 + 8);
            v18 = v31;
          }
          while ( v31 != v20 );
        }
        j_j_j___libc_free_0_0(v18 - 8);
      }
      goto LABEL_61;
    }
LABEL_31:
    if ( v12 == &v37 )
      goto LABEL_16;
    sub_127D120((_QWORD *)(v10 + 40));
    if ( v6 != v37 )
    {
      sub_1698450((__int64)v12, (__int64)&v37);
      goto LABEL_16;
    }
LABEL_61:
    sub_169C7E0(v12, &v37);
    goto LABEL_16;
  }
  if ( v6 == v37 )
    goto LABEL_31;
  sub_16983E0(v10 + 40, (__int64)&v37);
LABEL_16:
  result = sub_127D120(&v37);
  if ( v36 > 0x40 && v35 )
    result = j_j___libc_free_0_0(v35);
  if ( a2 )
    return sub_169C8D0(a1, a3, a4, a5);
  return result;
}
