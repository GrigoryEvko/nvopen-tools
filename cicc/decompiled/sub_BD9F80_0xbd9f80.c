// Function: sub_BD9F80
// Address: 0xbd9f80
//
__int64 __fastcall sub_BD9F80(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v8; // rbx
  __int64 i; // r9
  unsigned __int64 v10; // rsi
  __int64 v11; // r14
  unsigned __int8 v12; // dl
  unsigned __int64 v13; // r14
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // rsi
  unsigned __int8 v17; // dl
  __int64 v18; // rsi
  unsigned __int64 v19; // r12
  __int64 v20; // rcx
  unsigned __int8 v21; // dl
  __int64 v22; // r9
  __int64 v23; // r12
  unsigned __int8 v24; // dl
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // [rsp+0h] [rbp-30h]

  result = a3 & 1;
  v8 = (a3 - 1) / 2;
  v27 = result;
  if ( a2 >= v8 )
  {
    v14 = a1 + 8 * a2;
    if ( (a3 & 1) != 0 )
      goto LABEL_29;
    result = a2;
    goto LABEL_25;
  }
  for ( i = a2; ; i = result )
  {
    result = 2 * (i + 1);
    v14 = a1 + 16 * (i + 1);
    v15 = *(_QWORD *)v14;
    v16 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v14 - 32LL * (*(_DWORD *)(*(_QWORD *)v14 + 4LL) & 0x7FFFFFF)) + 24LL);
    v17 = *(_BYTE *)(v16 - 16);
    if ( (v17 & 2) != 0 )
      v10 = *(_QWORD *)(v16 - 32);
    else
      v10 = -16 - 8LL * ((v17 >> 2) & 0xF) + v16;
    v11 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v14 - 8) - 32LL * (*(_DWORD *)(*(_QWORD *)(v14 - 8) + 4LL) & 0x7FFFFFF))
                    + 24LL);
    v12 = *(_BYTE *)(v11 - 16);
    if ( (v12 & 2) != 0 )
      v13 = *(_QWORD *)(v11 - 32);
    else
      v13 = -16 - 8LL * ((v12 >> 2) & 0xF) + v11;
    if ( v13 > v10 )
    {
      --result;
      v14 = a1 + 8 * result;
      v15 = *(_QWORD *)v14;
    }
    *(_QWORD *)(a1 + 8 * i) = v15;
    if ( result >= v8 )
      break;
  }
  if ( !v27 )
  {
LABEL_25:
    if ( (a3 - 2) / 2 == result )
    {
      v25 = 2 * result + 2;
      v26 = *(_QWORD *)(a1 + 8 * v25 - 8);
      result = v25 - 1;
      *(_QWORD *)v14 = v26;
      v14 = a1 + 8 * result;
    }
  }
  v18 = (result - 1) / 2;
  if ( result <= a2 )
  {
LABEL_29:
    *(_QWORD *)v14 = a4;
    return result;
  }
  while ( 1 )
  {
    v22 = a1 + 8 * v18;
    v23 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v22 - 32LL * (*(_DWORD *)(*(_QWORD *)v22 + 4LL) & 0x7FFFFFF)) + 24LL);
    v24 = *(_BYTE *)(v23 - 16);
    v19 = (v24 & 2) != 0 ? *(_QWORD *)(v23 - 32) : -16 - 8LL * ((v24 >> 2) & 0xF) + v23;
    v20 = *(_QWORD *)(*(_QWORD *)(a4 - 32LL * (*(_DWORD *)(a4 + 4) & 0x7FFFFFF)) + 24LL);
    v21 = *(_BYTE *)(v20 - 16);
    if ( (v21 & 2) != 0 )
    {
      result = a1 + 8 * result;
      if ( *(_QWORD *)(v20 - 32) <= v19 )
        goto LABEL_28;
    }
    else
    {
      result = a1 + 8 * result;
      if ( -16LL - 8 * (unsigned __int64)((v21 >> 2) & 0xF) + v20 <= v19 )
      {
LABEL_28:
        v14 = result;
        goto LABEL_29;
      }
    }
    *(_QWORD *)result = *(_QWORD *)v22;
    result = v18;
    if ( a2 >= v18 )
      break;
    v18 = (v18 - 1) / 2;
  }
  *(_QWORD *)v22 = a4;
  return result;
}
