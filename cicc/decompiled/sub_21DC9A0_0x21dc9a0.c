// Function: sub_21DC9A0
// Address: 0x21dc9a0
//
__int64 __fastcall sub_21DC9A0(__int64 a1, __int64 a2, __int64 *a3, _DWORD *a4)
{
  __int64 v7; // r15
  __int64 v8; // r15
  unsigned __int16 v9; // ax
  __int64 v10; // r14
  __int64 v11; // rdx
  char *v12; // rsi
  const char *v13; // rax
  size_t v14; // rdx
  _WORD *v15; // rdi
  char *v16; // rsi
  unsigned __int64 v17; // rax
  void **v18; // r13
  __int64 result; // rax
  __int64 v20; // r14
  char *v21; // rax
  __int64 v22; // rax
  size_t v23; // [rsp+8h] [rbp-108h]
  unsigned int v24; // [rsp+20h] [rbp-F0h]
  _QWORD *v25; // [rsp+28h] [rbp-E8h]
  unsigned __int8 v26; // [rsp+28h] [rbp-E8h]
  __int64 v27; // [rsp+38h] [rbp-D8h] BYREF
  _QWORD *v28; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v29; // [rsp+48h] [rbp-C8h]
  _QWORD v30[2]; // [rsp+50h] [rbp-C0h] BYREF
  _QWORD v31[2]; // [rsp+60h] [rbp-B0h] BYREF
  _QWORD v32[2]; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v33[6]; // [rsp+80h] [rbp-90h] BYREF
  void *v34; // [rsp+B0h] [rbp-60h] BYREF
  void *v35; // [rsp+B8h] [rbp-58h]
  __int64 v36; // [rsp+C0h] [rbp-50h]
  void *dest; // [rsp+C8h] [rbp-48h]
  int v38; // [rsp+D0h] [rbp-40h]
  char **v39; // [rsp+D8h] [rbp-38h]

  v7 = a3[5];
  v25 = sub_21DC2B0((__int64)a3);
  v8 = sub_1E69D00(v7, *(_DWORD *)(a2 + 8));
  v9 = **(_WORD **)(v8 + 16);
  if ( v9 != 4878 )
  {
    if ( v9 > 0x130Eu )
    {
      if ( v9 == 4889 )
      {
        v20 = *(_QWORD *)(*(_QWORD *)(v8 + 32) + 64LL);
        v33[0] = v8;
        sub_21DC730((__int64)&v34, a1 + 232, v33);
        v21 = (char *)sub_1649960(v20);
        *a4 = sub_21DC560((__int64)v25, v21);
        return 1;
      }
      return 0;
    }
    if ( v9 != 15 )
    {
      if ( v9 == 3066 && *(_DWORD *)(a3[1] + 952) != 1 )
      {
        v10 = *(_QWORD *)(*(_QWORD *)(v8 + 32) + 264LL);
        v12 = (char *)sub_1E0A440(a3);
        if ( v12 )
        {
          v28 = v30;
          sub_21DBEB0((__int64 *)&v28, v12, (__int64)&v12[v11]);
          if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v29) <= 6 )
            sub_4262D8((__int64)"basic_string::append");
        }
        else
        {
          LOBYTE(v30[0]) = 0;
          v28 = v30;
          v29 = 0;
        }
        sub_2241490(&v28, "_param_", 7);
        v24 = strtol((const char *)(v10 + v29), 0, 10);
        v39 = (char **)v31;
        v31[0] = v32;
        v31[1] = 0;
        LOBYTE(v32[0]) = 0;
        v38 = 1;
        dest = 0;
        v36 = 0;
        v35 = 0;
        v34 = &unk_49EFBE0;
        v13 = sub_1E0A440(a3);
        v15 = dest;
        v16 = (char *)v13;
        v17 = v36 - (_QWORD)dest;
        if ( v36 - (__int64)dest < v14 )
        {
          v22 = sub_16E7EE0((__int64)&v34, v16, v14);
          v15 = *(_WORD **)(v22 + 24);
          v18 = (void **)v22;
          v17 = *(_QWORD *)(v22 + 16) - (_QWORD)v15;
        }
        else
        {
          v18 = &v34;
          if ( v14 )
          {
            v23 = v14;
            memcpy(dest, v16, v14);
            dest = (char *)dest + v23;
            v15 = dest;
            v17 = v36 - (_QWORD)dest;
          }
        }
        if ( v17 <= 6 )
        {
          v18 = (void **)sub_16E7EE0((__int64)v18, "_param_", 7u);
        }
        else
        {
          *(_DWORD *)v15 = 1918988383;
          v15[2] = 28001;
          *((_BYTE *)v15 + 6) = 95;
          v18[3] = (char *)v18[3] + 7;
        }
        sub_16E7A90((__int64)v18, v24);
        v27 = v8;
        sub_21DC730((__int64)v33, a1 + 232, &v27);
        if ( dest != v35 )
          sub_16E7BA0((__int64 *)&v34);
        *a4 = sub_21DC560((__int64)v25, *v39);
        sub_16E7BC0((__int64 *)&v34);
        if ( (_QWORD *)v31[0] != v32 )
          j_j___libc_free_0(v31[0], v32[0] + 1LL);
        if ( v28 != v30 )
          j_j___libc_free_0(v28, v30[0] + 1LL);
        return 1;
      }
      return 0;
    }
  }
  result = sub_21DC9A0(a1, *(_QWORD *)(v8 + 32) + 40LL, a3, a4);
  if ( (_BYTE)result )
  {
    v26 = result;
    v33[0] = v8;
    sub_21DC730((__int64)&v34, a1 + 232, v33);
    return v26;
  }
  return result;
}
