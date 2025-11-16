// Function: sub_BD8920
// Address: 0xbd8920
//
__int64 __fastcall sub_BD8920(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rax
  __int64 v5; // r14
  _QWORD *v6; // r15
  _QWORD *v7; // rbx
  unsigned int v8; // eax
  _QWORD *v9; // rdx
  const char *v10; // rax
  __int64 v11; // rdx
  const char *v12; // rax
  const char *v13; // r14
  size_t v14; // rdx
  _BYTE *v15; // rdi
  size_t v16; // rax
  _QWORD *v17; // rax
  __int64 v18; // rsi
  __int64 result; // rax
  size_t v20; // [rsp+0h] [rbp-160h]
  const char *v21; // [rsp+8h] [rbp-158h]
  size_t v22; // [rsp+8h] [rbp-158h]
  _BYTE *v23; // [rsp+10h] [rbp-150h] BYREF
  size_t v24; // [rsp+18h] [rbp-148h]
  __int64 v25; // [rsp+20h] [rbp-140h]
  _BYTE dest[312]; // [rsp+28h] [rbp-138h] BYREF

  v4 = (_QWORD *)sub_BD5C70(a2);
  v5 = *v4;
  v6 = v4 + 2;
  v7 = v4;
  v8 = sub_C92610(v4 + 2, *v4);
  v9 = (_QWORD *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_C92740(a1, v6, v5, v8));
  if ( !*v9 )
    goto LABEL_11;
  if ( *v9 == -8 )
  {
    --*(_DWORD *)(a1 + 16);
LABEL_11:
    *v9 = v7;
    ++*(_DWORD *)(a1 + 12);
    return sub_C929D0(a1, 0);
  }
  v10 = sub_BD5D20(a2);
  v21 = &v10[v11];
  v12 = sub_BD5D20(a2);
  v23 = dest;
  v24 = 0;
  v13 = v12;
  v25 = 256;
  v14 = v21 - v12;
  if ( (unsigned __int64)(v21 - v12) > 0x100 )
  {
    v20 = v21 - v12;
    sub_C8D290(&v23, dest, v14, 1);
    v16 = v24;
    v14 = v20;
    v15 = &v23[v24];
  }
  else
  {
    v15 = dest;
    v16 = 0;
  }
  if ( v13 != v21 )
  {
    v22 = v14;
    memcpy(v15, v13, v14);
    v16 = v24;
    v14 = v22;
  }
  v24 = v14 + v16;
  v17 = (_QWORD *)sub_BD5C70(a2);
  sub_C7D6A0(v17, *v17 + 17LL, 8);
  v18 = sub_BD8570(a1, a2, &v23);
  result = sub_BD6500(a2, v18);
  if ( v23 != dest )
    return _libc_free(v23, v18);
  return result;
}
