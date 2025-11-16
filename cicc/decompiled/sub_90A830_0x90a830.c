// Function: sub_90A830
// Address: 0x90a830
//
__int64 __fastcall sub_90A830(__int64 *a1, __int64 a2, const char *a3)
{
  const char *v3; // r12
  const void *v4; // r14
  size_t v5; // r13
  unsigned int v6; // eax
  __int64 v7; // r8
  __int64 v8; // r9
  _QWORD *v9; // rcx
  __int64 v10; // rbx
  __int64 result; // rax
  __int64 v12; // rax
  unsigned int v13; // r9d
  _QWORD *v14; // rcx
  _QWORD *v15; // rbx
  __int64 *v16; // rax
  __int64 v17; // rax
  __int64 v18; // r14
  bool v19; // zf
  __int64 v20; // r15
  __int64 v21; // r13
  _QWORD *v22; // [rsp+8h] [rbp-88h]
  unsigned int v23; // [rsp+14h] [rbp-7Ch]
  __int64 v24; // [rsp+18h] [rbp-78h]
  const char *v25; // [rsp+30h] [rbp-60h] BYREF
  __int16 v26; // [rsp+50h] [rbp-40h]

  v3 = a3;
  v4 = *(const void **)(a2 + 184);
  v5 = *(_QWORD *)(a2 + 176);
  if ( !a3 )
    v3 = "$str";
  v6 = sub_C92610(v4, v5);
  v8 = (unsigned int)sub_C92740(a1 + 57, v4, v5, v6);
  v9 = (_QWORD *)(a1[57] + 8 * v8);
  v10 = *v9;
  if ( *v9 )
  {
    if ( v10 != -8 )
      goto LABEL_5;
    --*((_DWORD *)a1 + 118);
  }
  v22 = v9;
  v23 = v8;
  v12 = sub_C7D670(v5 + 17, 8);
  v13 = v23;
  v14 = v22;
  v15 = (_QWORD *)v12;
  if ( v5 )
  {
    memcpy((void *)(v12 + 16), v4, v5);
    v13 = v23;
    v14 = v22;
  }
  *((_BYTE *)v15 + v5 + 16) = 0;
  *v15 = v5;
  v15[1] = 0;
  *v14 = v15;
  ++*((_DWORD *)a1 + 117);
  v16 = (__int64 *)(a1[57] + 8LL * (unsigned int)sub_C929D0(a1 + 57, v13));
  v10 = *v16;
  if ( *v16 && v10 != -8 )
  {
LABEL_5:
    result = *(_QWORD *)(v10 + 8);
    if ( result )
      return result;
    goto LABEL_15;
  }
  do
  {
    do
    {
      v10 = v16[1];
      ++v16;
    }
    while ( v10 == -8 );
  }
  while ( !v10 );
  result = *(_QWORD *)(v10 + 8);
  if ( !result )
  {
LABEL_15:
    v17 = sub_91FFA0(a1, a2, 0, v9, v7, v8);
    v18 = *a1;
    v19 = *v3 == 0;
    v20 = *(_QWORD *)(v17 + 8);
    v21 = v17;
    v26 = 257;
    if ( !v19 )
    {
      v25 = v3;
      LOBYTE(v26) = 3;
    }
    result = sub_BD2C40(88, unk_3F0FAE8);
    if ( result )
    {
      v24 = result;
      sub_B30000(result, v18, v20, 1, 8, v21, &v25, 0, 0, 0x100000001LL, 0);
      result = v24;
    }
    *(_QWORD *)(v10 + 8) = result;
  }
  return result;
}
