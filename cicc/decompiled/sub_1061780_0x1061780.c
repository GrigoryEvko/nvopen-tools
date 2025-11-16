// Function: sub_1061780
// Address: 0x1061780
//
__int64 __fastcall sub_1061780(__int64 **a1, const void *a2, unsigned __int64 a3, const void *a4, size_t a5)
{
  __int64 result; // rax
  __int64 v9; // rcx
  __int64 v10; // r9
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 *v13; // rsi
  __int64 *v14; // rax
  unsigned __int64 v15; // rdx
  _QWORD *v16; // rbx
  __int64 v17; // r12
  char *v18; // rdi
  __int64 v19; // r9
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // r15
  unsigned __int64 v22; // rdx
  __int64 *v24; // [rsp+10h] [rbp-150h] BYREF
  __int64 v25; // [rsp+18h] [rbp-148h]
  unsigned __int64 v26; // [rsp+20h] [rbp-140h]
  __int64 v27; // [rsp+28h] [rbp-138h] BYREF
  char v28; // [rsp+30h] [rbp-130h] BYREF

  result = sub_BA8B30(**a1, (__int64)a2, a3);
  if ( !result )
    return result;
  v26 = 256;
  v24 = &v27;
  v27 = 0x207265766D79732ELL;
  v25 = 8;
  if ( a3 + 8 > 0x100 )
  {
    sub_C8D290((__int64)&v24, &v27, a3 + 8, 1u, (__int64)&v24, v10);
    v18 = (char *)v24 + v25;
    goto LABEL_15;
  }
  v11 = 8;
  if ( a3 )
  {
    v18 = &v28;
LABEL_15:
    memcpy(v18, a2, a3);
    v11 = a3 + v25;
    v20 = a3 + v25 + 2;
    v25 += a3;
    if ( v20 > v26 )
    {
      sub_C8D290((__int64)&v24, &v27, v20, 1u, (__int64)&v24, v19);
      v11 = v25;
    }
  }
  *(_WORD *)((char *)v24 + v11) = 8236;
  v12 = v25 + 2;
  v25 = v12;
  if ( a5 + v12 > v26 )
  {
    sub_C8D290((__int64)&v24, &v27, a5 + v12, 1u, (__int64)&v24, a5 + v12);
    v12 = v25;
  }
  v13 = v24;
  if ( a5 )
  {
    memcpy((char *)v24 + v12, a4, a5);
    v13 = v24;
    v12 = v25;
  }
  v14 = *a1;
  v15 = a5 + v12;
  v25 = v15;
  v16 = (_QWORD *)*v14;
  if ( v15 > 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(*v14 + 96) )
    sub_4262D8((__int64)"basic_string::append");
  result = sub_2241490(v16 + 11, v13, v15, v9);
  v17 = v16[12];
  if ( v17 )
  {
    result = v16[11];
    if ( *(_BYTE *)(result + v17 - 1) != 10 )
    {
      v21 = v17 + 1;
      if ( (_QWORD *)result == v16 + 13 )
        v22 = 15;
      else
        v22 = v16[13];
      if ( v21 > v22 )
      {
        v13 = (__int64 *)v16[12];
        sub_2240BB0(v16 + 11, v13, 0, 0, 1);
        result = v16[11];
      }
      *(_BYTE *)(result + v17) = 10;
      result = v16[11];
      v16[12] = v21;
      *(_BYTE *)(result + v17 + 1) = 0;
    }
  }
  if ( v24 != &v27 )
    return _libc_free(v24, v13);
  return result;
}
