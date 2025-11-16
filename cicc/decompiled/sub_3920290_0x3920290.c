// Function: sub_3920290
// Address: 0x3920290
//
__int64 __fastcall sub_3920290(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 **v3; // r13
  __int64 **v4; // r12
  __int64 *v6; // rbx
  unsigned int v7; // esi
  __int64 v8; // r14
  __int64 v9; // r9
  unsigned int v10; // r8d
  _QWORD *v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rdx
  unsigned __int64 v14; // rdx
  int v15; // eax
  int v16; // esi
  __int64 v17; // rcx
  int v18; // edi
  _QWORD *v19; // rdx
  __int64 v20; // r8
  int v21; // r10d
  _QWORD *v22; // r9
  int v23; // r11d
  int v24; // eax
  int v25; // eax
  __int64 v26; // r8
  int v27; // r10d
  unsigned int v28; // ecx
  __int64 v29; // rsi
  unsigned int v30; // [rsp+0h] [rbp-70h]
  __int64 v31; // [rsp+8h] [rbp-68h]
  _QWORD v32[2]; // [rsp+10h] [rbp-60h] BYREF
  _QWORD v33[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v34; // [rsp+30h] [rbp-40h]

  result = a1 + 280;
  v3 = *(__int64 ***)(a2 + 64);
  v4 = *(__int64 ***)(a2 + 56);
  v31 = a1 + 280;
  while ( v3 != v4 )
  {
    v6 = *v4;
    if ( (**v4 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    {
      result = *((_BYTE *)v6 + 9) & 0xC;
      if ( (_BYTE)result != 8 )
        goto LABEL_4;
      *((_BYTE *)v6 + 8) |= 4u;
      v14 = (unsigned __int64)sub_38CE440(v6[3]);
      result = v14 | *v6 & 7;
      *v6 = result;
      if ( !v14 )
        goto LABEL_4;
    }
    if ( *((_DWORD *)v6 + 8) )
      goto LABEL_4;
    result = *((_BYTE *)v6 + 9) & 0xC;
    if ( (_BYTE)result == 8 )
      goto LABEL_4;
    if ( (*v6 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v7 = *(_DWORD *)(a1 + 304);
      v8 = *(_QWORD *)((*v6 & 0xFFFFFFFFFFFFFFF8LL) + 24);
      if ( !v7 )
        goto LABEL_16;
    }
    else
    {
      v7 = *(_DWORD *)(a1 + 304);
      v8 = 0;
      if ( !v7 )
      {
LABEL_16:
        ++*(_QWORD *)(a1 + 280);
        goto LABEL_17;
      }
    }
    v9 = *(_QWORD *)(a1 + 288);
    v10 = (v7 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v11 = (_QWORD *)(v9 + 16LL * v10);
    v12 = *v11;
    if ( v8 == *v11 )
    {
LABEL_11:
      v13 = *(_QWORD *)(v8 + 152);
      v32[1] = *(_QWORD *)(v8 + 160);
      v34 = 1283;
      v33[0] = "section already has a defining function: ";
      v32[0] = v13;
      v33[1] = v32;
      sub_16BCFB0((__int64)v33, 1u);
    }
    v23 = 1;
    v19 = 0;
    while ( v12 != -8 )
    {
      if ( v19 || v12 != -16 )
        v11 = v19;
      v10 = (v7 - 1) & (v23 + v10);
      v12 = *(_QWORD *)(v9 + 16LL * v10);
      if ( v8 == v12 )
        goto LABEL_11;
      ++v23;
      v19 = v11;
      v11 = (_QWORD *)(v9 + 16LL * v10);
    }
    if ( !v19 )
      v19 = v11;
    v24 = *(_DWORD *)(a1 + 296);
    ++*(_QWORD *)(a1 + 280);
    v18 = v24 + 1;
    if ( 4 * (v24 + 1) < 3 * v7 )
    {
      result = v7 - *(_DWORD *)(a1 + 300) - v18;
      if ( (unsigned int)result > v7 >> 3 )
        goto LABEL_30;
      v30 = ((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4);
      sub_39200D0(v31, v7);
      v25 = *(_DWORD *)(a1 + 304);
      if ( !v25 )
      {
LABEL_51:
        ++*(_DWORD *)(a1 + 296);
        BUG();
      }
      result = (unsigned int)(v25 - 1);
      v26 = *(_QWORD *)(a1 + 288);
      v22 = 0;
      v27 = 1;
      v28 = result & v30;
      v18 = *(_DWORD *)(a1 + 296) + 1;
      v19 = (_QWORD *)(v26 + 16LL * ((unsigned int)result & v30));
      v29 = *v19;
      if ( v8 == *v19 )
        goto LABEL_30;
      while ( v29 != -8 )
      {
        if ( v29 == -16 && !v22 )
          v22 = v19;
        v28 = result & (v27 + v28);
        v19 = (_QWORD *)(v26 + 16LL * v28);
        v29 = *v19;
        if ( v8 == *v19 )
          goto LABEL_30;
        ++v27;
      }
      goto LABEL_21;
    }
LABEL_17:
    sub_39200D0(v31, 2 * v7);
    v15 = *(_DWORD *)(a1 + 304);
    if ( !v15 )
      goto LABEL_51;
    v16 = v15 - 1;
    v17 = *(_QWORD *)(a1 + 288);
    result = (v15 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v18 = *(_DWORD *)(a1 + 296) + 1;
    v19 = (_QWORD *)(v17 + 16 * result);
    v20 = *v19;
    if ( *v19 == v8 )
      goto LABEL_30;
    v21 = 1;
    v22 = 0;
    while ( v20 != -8 )
    {
      if ( v20 == -16 && !v22 )
        v22 = v19;
      result = v16 & (unsigned int)(v21 + result);
      v19 = (_QWORD *)(v17 + 16LL * (unsigned int)result);
      v20 = *v19;
      if ( v8 == *v19 )
        goto LABEL_30;
      ++v21;
    }
LABEL_21:
    if ( v22 )
      v19 = v22;
LABEL_30:
    *(_DWORD *)(a1 + 296) = v18;
    if ( *v19 != -8 )
      --*(_DWORD *)(a1 + 300);
    *v19 = v8;
    v19[1] = v6;
LABEL_4:
    ++v4;
  }
  return result;
}
