// Function: sub_B58F60
// Address: 0xb58f60
//
__int64 __fastcall sub_B58F60(__int64 a1, int a2, unsigned __int8 *a3)
{
  __int64 v4; // rdi
  __int64 v6; // rdx
  __int64 v7; // rsi
  __int64 v8; // rax
  _BYTE *v9; // rcx
  __int64 result; // rax
  __int64 v11; // r12
  __int64 v12; // rax
  unsigned __int8 *v13; // r12
  unsigned int v14; // r15d
  unsigned __int8 *v15; // rax
  __int64 v16; // rdx
  unsigned __int64 v17; // r8
  int v18; // eax
  __int64 v19; // rdx
  unsigned int v20; // eax
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r12
  __int64 *v25; // r15
  __int64 *v26; // rax
  __int64 v27; // r12
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 v30; // r14
  __int64 v31; // rdx
  __int64 v32; // rdx
  unsigned __int8 *v33; // [rsp+8h] [rbp-68h]
  __int64 *v34; // [rsp+10h] [rbp-60h] BYREF
  __int64 v35; // [rsp+18h] [rbp-58h]
  _BYTE v36[80]; // [rsp+20h] [rbp-50h] BYREF

  v4 = (__int64)a3;
  v6 = *a3;
  v7 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  v8 = *(_QWORD *)v7;
  v9 = *(_BYTE **)(*(_QWORD *)v7 + 24LL);
  if ( *v9 != 4 )
  {
    if ( (_BYTE)v6 == 24 )
    {
      result = *(_QWORD *)(v7 + 8);
      **(_QWORD **)(v7 + 16) = result;
      if ( !result )
      {
        *(_QWORD *)v7 = v4;
LABEL_6:
        result = *(_QWORD *)(v4 + 16);
        *(_QWORD *)(v7 + 8) = result;
        if ( result )
          *(_QWORD *)(result + 16) = v7 + 8;
        *(_QWORD *)(v7 + 16) = v4 + 16;
        *(_QWORD *)(v4 + 16) = v7;
        return result;
      }
    }
    else
    {
      v11 = sub_B98A20(v4, v7, v6, v9);
      v12 = sub_BD5C60(a1, v7);
      v4 = sub_B9F6F0(v12, v11);
      result = 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
      v7 = a1 - result;
      if ( !*(_QWORD *)(a1 - result) || (result = *(_QWORD *)(v7 + 8), (**(_QWORD **)(v7 + 16) = result) == 0) )
      {
LABEL_5:
        *(_QWORD *)v7 = v4;
        if ( !v4 )
          return result;
        goto LABEL_6;
      }
    }
    *(_QWORD *)(result + 16) = *(_QWORD *)(v7 + 16);
    goto LABEL_5;
  }
  v34 = (__int64 *)v36;
  v35 = 0x400000000LL;
  if ( (_BYTE)v6 == 24 )
  {
    v13 = *(unsigned __int8 **)(v4 + 24);
    if ( (unsigned int)*v13 - 1 >= 2 )
      v13 = 0;
  }
  else
  {
    v13 = (unsigned __int8 *)sub_B98A20(v4, v7, v6, 0x400000000LL);
    v8 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  }
  v14 = 0;
  while ( 1 )
  {
    v19 = *(_QWORD *)(v8 + 24);
    v20 = 1;
    if ( *(_BYTE *)v19 == 4 )
      v20 = *(_DWORD *)(v19 + 144);
    if ( v14 >= v20 )
      break;
    v15 = v13;
    if ( v14 != a2 )
    {
      v7 = v14;
      v21 = sub_B58EB0(a1, v14);
      if ( *(_BYTE *)v21 != 24 )
      {
        v15 = (unsigned __int8 *)sub_B98A20(v21, v14, v22, v23);
        v16 = (unsigned int)v35;
        v17 = (unsigned int)v35 + 1LL;
        if ( v17 <= HIDWORD(v35) )
          goto LABEL_20;
        goto LABEL_27;
      }
      v15 = *(unsigned __int8 **)(v21 + 24);
      if ( (unsigned int)*v15 - 1 > 1 )
        v15 = 0;
    }
    v16 = (unsigned int)v35;
    v17 = (unsigned int)v35 + 1LL;
    if ( v17 <= HIDWORD(v35) )
      goto LABEL_20;
LABEL_27:
    v7 = (__int64)v36;
    v33 = v15;
    sub_C8D5F0(&v34, v36, v17, 8);
    v16 = (unsigned int)v35;
    v15 = v33;
LABEL_20:
    ++v14;
    v34[v16] = (__int64)v15;
    v18 = *(_DWORD *)(a1 + 4);
    LODWORD(v35) = v35 + 1;
    v8 = *(_QWORD *)(a1 - 32LL * (v18 & 0x7FFFFFF));
  }
  v24 = (unsigned int)v35;
  v25 = v34;
  v26 = (__int64 *)sub_BD5C60(a1, v7);
  v27 = sub_B00B60(v26, v25, v24);
  v28 = sub_BD5C60(a1, v25);
  v29 = v27;
  result = sub_B9F6F0(v28, v27);
  v30 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( *(_QWORD *)v30 )
  {
    v31 = *(_QWORD *)(v30 + 8);
    **(_QWORD **)(v30 + 16) = v31;
    if ( v31 )
      *(_QWORD *)(v31 + 16) = *(_QWORD *)(v30 + 16);
  }
  *(_QWORD *)v30 = result;
  if ( result )
  {
    v32 = *(_QWORD *)(result + 16);
    *(_QWORD *)(v30 + 8) = v32;
    if ( v32 )
    {
      v29 = v30 + 8;
      *(_QWORD *)(v32 + 16) = v30 + 8;
    }
    *(_QWORD *)(v30 + 16) = result + 16;
    *(_QWORD *)(result + 16) = v30;
  }
  if ( v34 != (__int64 *)v36 )
    return _libc_free(v34, v29);
  return result;
}
