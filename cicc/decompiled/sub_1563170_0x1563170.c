// Function: sub_1563170
// Address: 0x1563170
//
__int64 __fastcall sub_1563170(__int64 *a1, __int64 *a2, int a3, _BYTE *a4, size_t a5)
{
  __int64 result; // rax
  __int64 v9; // rbx
  const void *v10; // r8
  size_t v11; // r11
  __int64 v12; // r10
  __int64 *v13; // rdi
  int v14; // edx
  __int64 *v15; // rax
  size_t v16; // [rsp+0h] [rbp-80h]
  const void *v17; // [rsp+8h] [rbp-78h]
  int v18; // [rsp+10h] [rbp-70h]
  __int64 v19; // [rsp+10h] [rbp-70h]
  __int64 v20; // [rsp+18h] [rbp-68h]
  __int64 v21; // [rsp+18h] [rbp-68h]
  __int64 *v22; // [rsp+20h] [rbp-60h] BYREF
  __int64 v23; // [rsp+28h] [rbp-58h]
  _BYTE dest[80]; // [rsp+30h] [rbp-50h] BYREF

  if ( !sub_15602A0(a1, a3, a4, a5) )
    return *a1;
  v20 = (unsigned int)(a3 + 1);
  v9 = sub_15601B0(a1);
  v22 = (__int64 *)dest;
  v10 = (const void *)sub_15601A0(a1);
  v11 = v9 - (_QWORD)v10;
  v23 = 0x400000000LL;
  v12 = (v9 - (__int64)v10) >> 3;
  if ( (unsigned __int64)(v9 - (_QWORD)v10) > 0x20 )
  {
    v16 = v9 - (_QWORD)v10;
    v17 = v10;
    v19 = (v9 - (__int64)v10) >> 3;
    sub_16CD150(&v22, dest, v19, 8);
    v15 = v22;
    v14 = v23;
    LODWORD(v12) = v19;
    v10 = v17;
    v11 = v16;
    v13 = &v22[(unsigned int)v23];
  }
  else
  {
    v13 = (__int64 *)dest;
    v14 = 0;
    v15 = (__int64 *)dest;
  }
  if ( (const void *)v9 != v10 )
  {
    v18 = v12;
    memcpy(v13, v10, v11);
    v15 = v22;
    v14 = v23;
    LODWORD(v12) = v18;
  }
  LODWORD(v23) = v14 + v12;
  v15[v20] = sub_15630E0(&v15[v20], a2, a4, a5);
  result = sub_155F990(a2, v22, (unsigned int)v23);
  if ( v22 != (__int64 *)dest )
  {
    v21 = result;
    _libc_free((unsigned __int64)v22);
    return v21;
  }
  return result;
}
