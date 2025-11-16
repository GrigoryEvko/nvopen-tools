// Function: sub_DED850
// Address: 0xded850
//
__int64 __fastcall sub_DED850(__int64 a1, __int64 *a2, char *a3, __int64 a4, __int64 a5, unsigned __int8 a6)
{
  __int64 *v8; // rax
  __int64 *v11; // r15
  __int64 v12; // rdi
  unsigned int v13; // eax
  __int64 v14; // r9
  unsigned int v15; // ecx
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rsi
  __int64 v21; // rdx
  _QWORD *v22; // rax
  _QWORD *v23; // rax
  char **v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  char *v29; // rdi
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v42; // [rsp+10h] [rbp-80h] BYREF
  __int64 v43; // [rsp+18h] [rbp-78h]
  __int64 v44; // [rsp+20h] [rbp-70h]
  char v45; // [rsp+28h] [rbp-68h]
  char *v46; // [rsp+30h] [rbp-60h] BYREF
  int v47; // [rsp+38h] [rbp-58h]
  _BYTE v48[80]; // [rsp+40h] [rbp-50h] BYREF

  v8 = *(__int64 **)(a4 - 8);
  if ( v8[4] == a5 )
  {
    v35 = sub_D970F0((__int64)a2);
    sub_D97F80(a1, v35, v36, v37, v38, v39);
    return a1;
  }
  v11 = sub_DDFBA0((__int64)a2, *v8, a3);
  v12 = *(_QWORD *)(a4 - 8);
  if ( a5 != *(_QWORD *)(v12 + 32) )
  {
    v13 = (*(_DWORD *)(a4 + 4) & 0x7FFFFFFu) >> 1;
    v14 = v13 - 1;
    if ( v13 != 1 )
    {
      v15 = 2;
      v16 = 1;
      v17 = 0;
      while ( 1 )
      {
        while ( 1 )
        {
          v19 = 32;
          if ( (_DWORD)v16 != -1 )
            v19 = 32LL * (v15 + 1);
          if ( a5 == *(_QWORD *)(v12 + v19) )
            break;
          v18 = v16;
          v15 += 2;
          ++v16;
          if ( v14 == v18 )
            goto LABEL_11;
        }
        if ( v17 )
          break;
        v20 = v15;
        v21 = v16;
        v15 += 2;
        ++v16;
        v17 = *(_QWORD *)(v12 + 32 * v20);
        if ( v14 == v21 )
          goto LABEL_11;
      }
    }
  }
  v17 = 0;
LABEL_11:
  v22 = sub_DA2570((__int64)a2, v17);
  v23 = sub_DCC810(a2, (__int64)v11, (__int64)v22, 0, 0);
  v24 = (char **)a2;
  sub_DEC310((__int64)&v42, a2, (__int64)v23, (char **)a3, a6, 0);
  if ( sub_D96A50(v42) && sub_D96A50(v43) )
  {
    v24 = (char **)sub_D970F0((__int64)a2);
    sub_D97F80(a1, (__int64)v24, v31, v32, v33, v34);
    v29 = v46;
    if ( v46 == v48 )
      return a1;
    goto LABEL_14;
  }
  *(_QWORD *)a1 = v42;
  *(_QWORD *)(a1 + 8) = v43;
  *(_QWORD *)(a1 + 16) = v44;
  *(_BYTE *)(a1 + 24) = v45;
  *(_QWORD *)(a1 + 32) = a1 + 48;
  *(_QWORD *)(a1 + 40) = 0x400000000LL;
  if ( v47 )
  {
    v24 = &v46;
    sub_D91460(a1 + 32, &v46, v25, v26, v27, v28);
    v29 = v46;
    if ( v46 == v48 )
      return a1;
    goto LABEL_14;
  }
  v29 = v46;
  if ( v46 != v48 )
LABEL_14:
    _libc_free(v29, v24);
  return a1;
}
