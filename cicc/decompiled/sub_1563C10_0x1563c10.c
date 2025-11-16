// Function: sub_1563C10
// Address: 0x1563c10
//
__int64 __fastcall sub_1563C10(__int64 *a1, __int64 *a2, int a3, int a4)
{
  __int64 result; // rax
  __int64 v7; // rax
  const void *v8; // r9
  const void *v9; // r12
  size_t v10; // r11
  __int64 v11; // r10
  __int64 *v12; // rdi
  int v13; // eax
  __int64 *v14; // rbx
  size_t v15; // [rsp+0h] [rbp-80h]
  __int64 v16; // [rsp+8h] [rbp-78h]
  __int64 v17; // [rsp+10h] [rbp-70h]
  int v18; // [rsp+10h] [rbp-70h]
  __int64 v19; // [rsp+10h] [rbp-70h]
  __int64 v20; // [rsp+18h] [rbp-68h]
  __int64 v21; // [rsp+18h] [rbp-68h]
  __int64 *v22; // [rsp+20h] [rbp-60h] BYREF
  __int64 v23; // [rsp+28h] [rbp-58h]
  _BYTE dest[80]; // [rsp+30h] [rbp-50h] BYREF

  if ( !(unsigned __int8)sub_1560260(a1, a3, a4) )
    return *a1;
  v20 = (unsigned int)(a3 + 1);
  v17 = sub_15601B0(a1);
  v7 = sub_15601A0(a1);
  v8 = (const void *)v17;
  v22 = (__int64 *)dest;
  v9 = (const void *)v7;
  v23 = 0x400000000LL;
  v10 = v17 - v7;
  v11 = (v17 - v7) >> 3;
  if ( (unsigned __int64)(v17 - v7) > 0x20 )
  {
    v15 = v17 - v7;
    v16 = v17;
    v19 = (v17 - v7) >> 3;
    sub_16CD150(&v22, dest, v19, 8);
    v14 = v22;
    v13 = v23;
    LODWORD(v11) = v19;
    v8 = (const void *)v16;
    v10 = v15;
    v12 = &v22[(unsigned int)v23];
  }
  else
  {
    v12 = (__int64 *)dest;
    v13 = 0;
    v14 = (__int64 *)dest;
  }
  if ( v8 != v9 )
  {
    v18 = v11;
    memcpy(v12, v9, v10);
    v14 = v22;
    v13 = v23;
    LODWORD(v11) = v18;
  }
  LODWORD(v23) = v11 + v13;
  v14[v20] = sub_1563B90(&v14[v20], a2, a4);
  result = sub_155F990(a2, v22, (unsigned int)v23);
  if ( v22 != (__int64 *)dest )
  {
    v21 = result;
    _libc_free((unsigned __int64)v22);
    return v21;
  }
  return result;
}
