// Function: sub_1218310
// Address: 0x1218310
//
__int64 __fastcall sub_1218310(__int64 a1)
{
  __int64 v1; // r14
  unsigned __int64 v3; // r13
  int v4; // eax
  unsigned __int64 v5; // rsi
  __int64 result; // rax
  unsigned int v7; // eax
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rbx
  const char *v11; // rax
  unsigned __int8 v12; // [rsp+Fh] [rbp-C1h]
  unsigned int v13; // [rsp+14h] [rbp-BCh] BYREF
  unsigned __int64 v14; // [rsp+18h] [rbp-B8h] BYREF
  _QWORD v15[2]; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v16; // [rsp+30h] [rbp-A0h]
  const char *v17; // [rsp+40h] [rbp-90h] BYREF
  char *v18; // [rsp+48h] [rbp-88h]
  __int64 v19; // [rsp+50h] [rbp-80h]
  char v20; // [rsp+58h] [rbp-78h] BYREF
  char v21; // [rsp+60h] [rbp-70h]
  char v22; // [rsp+61h] [rbp-6Fh]

  v1 = a1 + 176;
  v3 = *(_QWORD *)(a1 + 232);
  v4 = sub_1205200(a1 + 176);
  *(_DWORD *)(a1 + 240) = v4;
  if ( v4 != 505 )
  {
    v5 = *(_QWORD *)(a1 + 232);
    v17 = "expected attribute group id";
    v22 = 1;
    v21 = 3;
    sub_11FD800(v1, v5, (__int64)&v17, 1);
    return 1;
  }
  v7 = *(_DWORD *)(a1 + 280);
  v15[0] = 0;
  v15[1] = 0;
  v13 = v7;
  v16 = 0;
  v14 = 0;
  *(_DWORD *)(a1 + 240) = sub_1205200(v1);
  if ( (unsigned __int8)sub_120AFE0(a1, 3, "expected '=' here")
    || (unsigned __int8)sub_120AFE0(a1, 8, "expected '{' here") )
  {
    goto LABEL_5;
  }
  v8 = *(_QWORD *)(a1 + 1496);
  v9 = a1 + 1488;
  if ( !v8 )
    goto LABEL_17;
  v10 = a1 + 1488;
  do
  {
    if ( *(_DWORD *)(v8 + 32) < v13 )
    {
      v8 = *(_QWORD *)(v8 + 24);
    }
    else
    {
      v10 = v8;
      v8 = *(_QWORD *)(v8 + 16);
    }
  }
  while ( v8 );
  if ( v10 == v9 || v13 < *(_DWORD *)(v10 + 32) )
  {
LABEL_17:
    v11 = **(const char ***)(a1 + 344);
    v18 = &v20;
    v17 = v11;
    v19 = 0x800000000LL;
    v10 = sub_1213000((_QWORD *)(a1 + 1480), &v13, (__int64 *)&v17);
    if ( v18 != &v20 )
      _libc_free(v18, &v13);
  }
  if ( (unsigned __int8)sub_1218010(a1, (__int64 **)(v10 + 40), (__int64)v15, 1u, &v14) )
    goto LABEL_5;
  result = sub_120AFE0(a1, 9, "expected end of attribute group");
  if ( (_BYTE)result )
    goto LABEL_5;
  if ( !*(_DWORD *)(v10 + 56) )
  {
    v22 = 1;
    v21 = 3;
    v17 = "attribute group has no attributes";
    sub_11FD800(v1, v3, (__int64)&v17, 1);
LABEL_5:
    result = 1;
  }
  if ( v15[0] )
  {
    v12 = result;
    j_j___libc_free_0(v15[0], v16 - v15[0]);
    return v12;
  }
  return result;
}
