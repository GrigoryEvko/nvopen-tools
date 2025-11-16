// Function: sub_D97D90
// Address: 0xd97d90
//
__int64 __fastcall sub_D97D90(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        char a5,
        __int64 a6,
        __int64 **a7,
        __int64 a8)
{
  __int64 **v10; // r12
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 result; // rax
  char v14; // di
  __int64 v15; // rbx
  __int64 *v16; // r14
  __int64 v17; // r13
  char *v18; // rdx
  __int64 v19; // rcx
  const void *v20; // [rsp+8h] [rbp-88h]
  __int64 v21; // [rsp+20h] [rbp-70h] BYREF
  char *v22; // [rsp+28h] [rbp-68h]
  __int64 v23; // [rsp+30h] [rbp-60h]
  int v24; // [rsp+38h] [rbp-58h]
  char v25; // [rsp+3Ch] [rbp-54h]
  char v26; // [rsp+40h] [rbp-50h] BYREF

  *(_QWORD *)(a1 + 32) = a1 + 48;
  v10 = a7;
  v20 = (const void *)(a1 + 48);
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = a3;
  *(_QWORD *)(a1 + 16) = a4;
  *(_BYTE *)(a1 + 24) = a5;
  *(_QWORD *)(a1 + 40) = 0x400000000LL;
  if ( sub_D968A0(a3) )
  {
    *(_QWORD *)a1 = a3;
    *(_QWORD *)(a1 + 16) = a3;
  }
  v25 = 1;
  v22 = &v26;
  result = (__int64)&a7[2 * a8];
  v14 = 1;
  v21 = 0;
  v23 = 4;
  v24 = 0;
  if ( a7 != (__int64 **)result )
  {
    while ( 1 )
    {
      result = (__int64)v10[1];
      v15 = (__int64)&(*v10)[result];
      if ( *v10 != (__int64 *)v15 )
        break;
LABEL_12:
      v10 += 2;
      if ( &a7[2 * a8] == v10 )
      {
        if ( !v14 )
          return _libc_free(v22, a2);
        return result;
      }
    }
    v16 = *v10;
    while ( 1 )
    {
      while ( 1 )
      {
        v17 = *v16;
        if ( !v14 )
          break;
        v18 = v22;
        v19 = (__int64)&v22[8 * HIDWORD(v23)];
        a2 = HIDWORD(v23);
        result = (__int64)v22;
        if ( v22 == (char *)v19 )
          goto LABEL_27;
        while ( v17 != *(_QWORD *)result )
        {
          result += 8;
          if ( v19 == result )
            goto LABEL_17;
        }
        if ( (__int64 *)v15 == ++v16 )
          goto LABEL_12;
      }
      a2 = *v16;
      result = (__int64)sub_C8CA60((__int64)&v21, *v16);
      if ( !result )
        break;
LABEL_21:
      ++v16;
      v14 = v25;
      if ( (__int64 *)v15 == v16 )
        goto LABEL_12;
    }
    if ( !v25 )
      goto LABEL_25;
    v18 = v22;
    a2 = HIDWORD(v23);
    result = (__int64)&v22[8 * HIDWORD(v23)];
    if ( v22 != (char *)result )
    {
LABEL_17:
      while ( v17 != *(_QWORD *)v18 )
      {
        v18 += 8;
        if ( v18 == (char *)result )
          goto LABEL_27;
      }
      goto LABEL_18;
    }
LABEL_27:
    if ( (unsigned int)v23 > (unsigned int)a2 )
    {
      a2 = (unsigned int)(a2 + 1);
      HIDWORD(v23) = a2;
      *(_QWORD *)result = v17;
      ++v21;
    }
    else
    {
LABEL_25:
      a2 = v17;
      sub_C8CC70((__int64)&v21, v17, (__int64)v18, v19, v11, v12);
    }
LABEL_18:
    result = *(unsigned int *)(a1 + 40);
    if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
    {
      a2 = (__int64)v20;
      sub_C8D5F0(a1 + 32, v20, result + 1, 8u, v11, v12);
      result = *(unsigned int *)(a1 + 40);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * result) = v17;
    ++*(_DWORD *)(a1 + 40);
    goto LABEL_21;
  }
  return result;
}
