// Function: sub_126A1B0
// Address: 0x126a1b0
//
__int64 __fastcall sub_126A1B0(__int64 *a1, __int64 a2, const char *a3)
{
  const char *v5; // r12
  size_t v6; // r15
  void *v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rsi
  __int64 v11; // r8
  __int64 v12; // r9
  _QWORD *v13; // r10
  __int64 v14; // rbx
  __int64 result; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  unsigned int v19; // r9d
  _QWORD *v20; // r10
  _QWORD *v21; // r8
  void *v22; // rdi
  __int64 *v23; // rax
  __int64 *v24; // rax
  __int64 *v25; // rax
  bool v26; // zf
  __int64 v27; // r14
  __int64 v28; // r15
  int v29; // r13d
  __int64 v30; // rax
  void *v31; // rax
  _QWORD *v32; // [rsp+0h] [rbp-80h]
  _QWORD *v33; // [rsp+8h] [rbp-78h]
  _QWORD *v34; // [rsp+8h] [rbp-78h]
  _QWORD *v35; // [rsp+8h] [rbp-78h]
  _QWORD *v36; // [rsp+10h] [rbp-70h]
  unsigned int v37; // [rsp+10h] [rbp-70h]
  unsigned int v38; // [rsp+18h] [rbp-68h]
  void *src; // [rsp+20h] [rbp-60h]
  __int64 *v40; // [rsp+28h] [rbp-58h]
  __int64 v41; // [rsp+28h] [rbp-58h]
  const char *v42; // [rsp+30h] [rbp-50h] BYREF
  __int16 v43; // [rsp+40h] [rbp-40h]

  v5 = a3;
  v6 = *(_QWORD *)(a2 + 176);
  v7 = *(void **)(a2 + 184);
  if ( !a3 )
    v5 = "$str";
  src = v7;
  v40 = a1 + 59;
  v10 = (unsigned int)sub_16D19C0(a1 + 59, v7, v6);
  v12 = v10;
  v13 = (_QWORD *)(a1[59] + 8 * v10);
  v14 = *v13;
  if ( *v13 )
  {
    if ( v14 != -8 )
      goto LABEL_5;
    --*((_DWORD *)a1 + 122);
  }
  v33 = v13;
  v16 = malloc(v6 + 17, v10, v8, v9, v11, v10);
  v19 = v10;
  v20 = v33;
  v21 = (_QWORD *)v16;
  if ( !v16 )
  {
    if ( v6 == -17 )
    {
      v30 = malloc(1, v10, v17, v18, 0, (unsigned int)v10);
      v19 = v10;
      v20 = v33;
      v21 = 0;
      if ( v30 )
      {
        v22 = (void *)(v30 + 16);
        v21 = (_QWORD *)v30;
        goto LABEL_23;
      }
    }
    v32 = v21;
    v35 = v20;
    v37 = v19;
    sub_16BD1C0("Allocation failed");
    v19 = v37;
    v20 = v35;
    v21 = v32;
  }
  v22 = v21 + 2;
  if ( v6 + 1 > 1 )
  {
LABEL_23:
    v34 = v21;
    v36 = v20;
    v38 = v19;
    v31 = memcpy(v22, src, v6);
    v21 = v34;
    v20 = v36;
    v19 = v38;
    v22 = v31;
  }
  *((_BYTE *)v22 + v6) = 0;
  *v21 = v6;
  v21[1] = 0;
  *v20 = v21;
  ++*((_DWORD *)a1 + 121);
  v23 = (__int64 *)(a1[59] + 8LL * (unsigned int)sub_16D1CD0(v40, v19));
  v14 = *v23;
  if ( !*v23 || v14 == -8 )
  {
    v24 = v23 + 1;
    do
    {
      do
        v14 = *v24++;
      while ( !v14 );
    }
    while ( v14 == -8 );
    result = *(_QWORD *)(v14 + 8);
    if ( result )
      return result;
    goto LABEL_16;
  }
LABEL_5:
  result = *(_QWORD *)(v14 + 8);
  if ( result )
    return result;
LABEL_16:
  v25 = (__int64 *)sub_127F610(a1, a2, 0, v9, v11, v12);
  v26 = *v5 == 0;
  v27 = *a1;
  v28 = *v25;
  v29 = (int)v25;
  v43 = 257;
  if ( !v26 )
  {
    v42 = v5;
    LOBYTE(v43) = 3;
  }
  result = sub_1648A60(88, 1);
  if ( result )
  {
    v41 = result;
    sub_15E51E0(result, v27, v28, 1, 8, v29, (__int64)&v42, 0, 0, 1, 0);
    result = v41;
  }
  *(_QWORD *)(v14 + 8) = result;
  return result;
}
