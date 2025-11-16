// Function: sub_1560530
// Address: 0x1560530
//
__int64 __fastcall sub_1560530(__int64 *a1, __int64 *a2, int a3)
{
  __int64 v3; // rbx
  __int64 result; // rax
  __int64 v5; // r15
  const void *v6; // r12
  size_t v7; // r9
  __int64 v8; // r8
  __int64 *v9; // rdi
  int v10; // eax
  __int64 *v11; // rdx
  int v12; // [rsp-70h] [rbp-70h]
  __int64 v13; // [rsp-70h] [rbp-70h]
  __int64 *v14; // [rsp-68h] [rbp-68h] BYREF
  __int64 v15; // [rsp-60h] [rbp-60h]
  _BYTE v16[88]; // [rsp-58h] [rbp-58h] BYREF

  if ( !*a1 )
    return 0;
  v3 = (unsigned int)(a3 + 1);
  if ( a3 == -1 )
    v3 = 0;
  if ( (unsigned int)sub_15601D0((__int64)a1) <= (unsigned int)v3 )
    return *a1;
  v5 = sub_15601B0(a1);
  v14 = (__int64 *)v16;
  v6 = (const void *)sub_15601A0(a1);
  v7 = v5 - (_QWORD)v6;
  v15 = 0x400000000LL;
  v8 = (v5 - (__int64)v6) >> 3;
  if ( (unsigned __int64)(v5 - (_QWORD)v6) > 0x20 )
  {
    sub_16CD150(&v14, v16, (v5 - (__int64)v6) >> 3, 8);
    v11 = v14;
    v10 = v15;
    v8 = (v5 - (__int64)v6) >> 3;
    v7 = v5 - (_QWORD)v6;
    v9 = &v14[(unsigned int)v15];
  }
  else
  {
    v9 = (__int64 *)v16;
    v10 = 0;
    v11 = (__int64 *)v16;
  }
  if ( (const void *)v5 != v6 )
  {
    v12 = v8;
    memcpy(v9, v6, v7);
    v11 = v14;
    v10 = v15;
    LODWORD(v8) = v12;
  }
  LODWORD(v15) = v8 + v10;
  v11[v3] = 0;
  result = sub_155F990(a2, v14, (unsigned int)v15);
  if ( v14 != (__int64 *)v16 )
  {
    v13 = result;
    _libc_free((unsigned __int64)v14);
    return v13;
  }
  return result;
}
