// Function: sub_1E51780
// Address: 0x1e51780
//
__int64 __fastcall sub_1E51780(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v6; // rbx
  __int64 v7; // r14
  unsigned __int64 v8; // rdi
  int v9; // eax
  int v10; // edx
  __int64 v11; // rsi
  unsigned int v12; // eax
  __int64 v13; // rcx
  __int64 v14; // rbx
  __int64 v15; // r14
  unsigned __int64 v16; // rdi
  int v17; // eax
  int v18; // edx
  __int64 v19; // rsi
  __int64 v20; // rcx
  int v21; // r8d
  int v22; // r8d
  __int64 v23; // [rsp+8h] [rbp-38h] BYREF
  __int64 v24[5]; // [rsp+18h] [rbp-28h] BYREF

  v23 = a1;
  v24[0] = a1;
  sub_1E51470(a2, v24);
  sub_1E51470(a3, &v23);
  result = v23;
  v6 = *(_QWORD *)(v23 + 112);
  v7 = v6 + 16LL * *(unsigned int *)(v23 + 120);
  if ( v6 != v7 )
  {
    while ( 1 )
    {
      v8 = *(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL;
      if ( ((*(_BYTE *)v6 ^ 6) & 6) == 0 && *(_DWORD *)(v6 + 8) == 3 )
        goto LABEL_5;
      v9 = *(_DWORD *)(a3 + 24);
      if ( !v9 )
        goto LABEL_15;
      v10 = v9 - 1;
      v11 = *(_QWORD *)(a3 + 8);
      v12 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v13 = *(_QWORD *)(v11 + 8LL * v12);
      if ( v8 != v13 )
        break;
LABEL_5:
      v6 += 16;
      if ( v7 == v6 )
      {
        result = v23;
        goto LABEL_7;
      }
    }
    v21 = 1;
    while ( v13 != -8 )
    {
      v12 = v10 & (v21 + v12);
      v13 = *(_QWORD *)(v11 + 8LL * v12);
      if ( v8 == v13 )
        goto LABEL_5;
      ++v21;
    }
LABEL_15:
    sub_1E51780(v8, a2, a3);
    goto LABEL_5;
  }
LABEL_7:
  v14 = *(_QWORD *)(result + 32);
  v15 = v14 + 16LL * *(unsigned int *)(result + 40);
  if ( v14 != v15 )
  {
    while ( 1 )
    {
      result = *(_QWORD *)v14 ^ 6LL;
      v16 = *(_QWORD *)v14 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (result & 6) == 0 && *(_DWORD *)(v14 + 8) == 3 )
        goto LABEL_11;
      v17 = *(_DWORD *)(a3 + 24);
      if ( !v17 )
        goto LABEL_18;
      v18 = v17 - 1;
      v19 = *(_QWORD *)(a3 + 8);
      result = (v17 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v20 = *(_QWORD *)(v19 + 8 * result);
      if ( v16 != v20 )
        break;
LABEL_11:
      v14 += 16;
      if ( v15 == v14 )
        return result;
    }
    v22 = 1;
    while ( v20 != -8 )
    {
      result = v18 & (unsigned int)(v22 + result);
      v20 = *(_QWORD *)(v19 + 8LL * (unsigned int)result);
      if ( v16 == v20 )
        goto LABEL_11;
      ++v22;
    }
LABEL_18:
    result = sub_1E51780(v16, a2, a3);
    goto LABEL_11;
  }
  return result;
}
