// Function: sub_E83A00
// Address: 0xe83a00
//
__int64 __fastcall sub_E83A00(__int64 a1, unsigned __int8 **a2, __int64 a3)
{
  unsigned __int64 v3; // rdx
  unsigned __int8 **v4; // r15
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 result; // rax
  unsigned __int64 v8; // r13
  unsigned __int8 **v9; // r12
  _QWORD *v10; // rbx
  unsigned __int8 *v11; // r8
  size_t v12; // r13
  _BYTE *v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rsi
  _QWORD *v16; // r12
  unsigned __int8 *v17; // [rsp+8h] [rbp-68h]
  size_t v18; // [rsp+18h] [rbp-58h] BYREF
  _QWORD *v19; // [rsp+20h] [rbp-50h] BYREF
  _QWORD *v20; // [rsp+28h] [rbp-48h]
  __int64 v21; // [rsp+30h] [rbp-40h]

  v3 = 32 * a3;
  v4 = &a2[v3 / 8];
  v5 = *(_QWORD *)(a1 + 296);
  v19 = 0;
  v20 = 0;
  v6 = *(_QWORD *)(v5 + 24);
  v21 = 0;
  result = 0x7FFFFFFFFFFFFFE0LL;
  if ( v3 > 0x7FFFFFFFFFFFFFE0LL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v8 = v3;
  if ( v3 )
  {
    v9 = a2;
    result = sub_22077B0(v3);
    v19 = (_QWORD *)result;
    v10 = (_QWORD *)result;
    v21 = result + v8;
    if ( a2 != v4 )
    {
      result = (__int64)&v18;
      while ( 1 )
      {
        if ( !v10 )
          goto LABEL_7;
        v11 = *v9;
        v12 = (size_t)v9[1];
        v13 = v10 + 2;
        *v10 = v10 + 2;
        result = (__int64)&v11[v12];
        if ( &v11[v12] && !v11 )
          sub_426248((__int64)"basic_string::_M_construct null not valid");
        v18 = v12;
        if ( v12 > 0xF )
          break;
        if ( v12 == 1 )
        {
          result = *v11;
          *((_BYTE *)v10 + 16) = result;
        }
        else if ( v12 )
        {
          goto LABEL_16;
        }
LABEL_6:
        v10[1] = v12;
        v13[v12] = 0;
LABEL_7:
        v9 += 4;
        v10 += 4;
        if ( v4 == v9 )
          goto LABEL_18;
      }
      v17 = v11;
      v14 = sub_22409D0(v10, &v18, 0);
      v11 = v17;
      *v10 = v14;
      v13 = (_BYTE *)v14;
      v10[2] = v18;
LABEL_16:
      result = (__int64)memcpy(v13, v11, v12);
      v12 = v18;
      v13 = (_BYTE *)*v10;
      goto LABEL_6;
    }
  }
  else
  {
    v10 = 0;
  }
LABEL_18:
  v20 = v10;
  v15 = *(_QWORD *)(v6 + 2032);
  if ( v15 == *(_QWORD *)(v6 + 2040) )
  {
    result = (__int64)sub_E83790((char **)(v6 + 2024), (char *)v15, &v19);
    v10 = v20;
    v16 = v19;
  }
  else
  {
    if ( v15 )
    {
      *(_QWORD *)v15 = v19;
      *(_QWORD *)(v15 + 8) = v20;
      result = v21;
      *(_QWORD *)(v15 + 16) = v21;
      *(_QWORD *)(v6 + 2032) += 24LL;
      return result;
    }
    *(_QWORD *)(v6 + 2032) = 24;
    v16 = v19;
  }
  if ( v10 != v16 )
  {
    do
    {
      result = (__int64)(v16 + 2);
      if ( (_QWORD *)*v16 != v16 + 2 )
        result = j_j___libc_free_0(*v16, v16[2] + 1LL);
      v16 += 4;
    }
    while ( v10 != v16 );
    v16 = v19;
  }
  if ( v16 )
    return j_j___libc_free_0(v16, v21 - (_QWORD)v16);
  return result;
}
