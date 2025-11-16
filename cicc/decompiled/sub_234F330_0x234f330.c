// Function: sub_234F330
// Address: 0x234f330
//
__int64 __fastcall sub_234F330(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdx
  int v5; // eax
  __int64 v6; // rdx
  __int64 v7; // rdx
  int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // rsi
  __int64 v11; // rax
  _QWORD *v12; // rax
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // r12
  __int64 result; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  _QWORD v19[6]; // [rsp+8h] [rbp-78h] BYREF
  _QWORD v20[9]; // [rsp+38h] [rbp-48h] BYREF

  v3 = a1 + 88;
  *(_QWORD *)(v3 - 80) = 0;
  *(_QWORD *)(v3 - 72) = 0;
  *(_DWORD *)(v3 - 64) = 0;
  v4 = *(_QWORD *)(a2 + 8);
  v5 = *(_DWORD *)(a2 + 24);
  ++*(_QWORD *)a2;
  *(_QWORD *)(v3 - 80) = v4;
  v6 = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(a2 + 8) = 0;
  *(_QWORD *)(a2 + 16) = 0;
  *(_DWORD *)(a2 + 24) = 0;
  *(_QWORD *)(v3 - 72) = v6;
  *(_QWORD *)(v3 - 48) = 0;
  *(_QWORD *)(v3 - 40) = 0;
  *(_DWORD *)(v3 - 32) = 0;
  v7 = *(_QWORD *)(a2 + 40);
  *(_DWORD *)(v3 - 64) = v5;
  v8 = *(_DWORD *)(a2 + 56);
  *(_QWORD *)(v3 - 48) = v7;
  v9 = *(_QWORD *)(a2 + 48);
  ++*(_QWORD *)(a2 + 32);
  v10 = a2 + 88;
  *(_QWORD *)(v3 - 40) = v9;
  *(_QWORD *)(v3 - 88) = 1;
  *(_QWORD *)(v3 - 56) = 1;
  *(_DWORD *)(v3 - 32) = v8;
  v11 = *(_QWORD *)(v10 - 24);
  *(_QWORD *)(v10 - 48) = 0;
  *(_QWORD *)(v3 - 24) = v11;
  v12 = (_QWORD *)(a1 + 104);
  *(_QWORD *)(v10 - 40) = 0;
  *(_DWORD *)(v10 - 32) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 1;
  do
  {
    if ( v12 )
      *v12 = -4096;
    v12 += 2;
  }
  while ( v12 != (_QWORD *)(a1 + 168) );
  sub_234F1D0(v3, v10);
  v13 = a1 + 184;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 1;
  do
  {
    if ( v13 )
    {
      *(_QWORD *)v13 = -4096;
      *(_DWORD *)(v13 + 8) = 0x7FFFFFFF;
    }
    v13 += 24;
  }
  while ( v13 != a1 + 280 );
  v14 = *(_QWORD *)(a1 + 8);
  v15 = v14 + 40LL * *(unsigned int *)(a1 + 24);
  result = *(unsigned int *)(a1 + 16);
  if ( (_DWORD)result )
  {
    v19[0] = 2;
    v19[1] = 0;
    v19[2] = -4096;
    v19[3] = 0;
    v20[0] = 2;
    v20[1] = 0;
    v20[2] = -8192;
    for ( v20[3] = 0; v15 != v14; v14 += 40 )
    {
      v17 = *(_QWORD *)(v14 + 24);
      if ( v17 != -4096 && v17 != -8192 )
        break;
    }
    v19[5] = &unk_49DB368;
    sub_D68D70(v20);
    sub_D68D70(v19);
    result = *(_QWORD *)(a1 + 8);
    v18 = result + 40LL * *(unsigned int *)(a1 + 24);
    while ( v18 != v14 )
    {
      *(_QWORD *)(v14 + 32) = a1;
      for ( v14 += 40; v15 != v14; v14 += 40 )
      {
        result = *(_QWORD *)(v14 + 24);
        if ( result != -8192 && result != -4096 )
          break;
      }
    }
  }
  return result;
}
