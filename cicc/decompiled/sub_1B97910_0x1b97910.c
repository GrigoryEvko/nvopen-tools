// Function: sub_1B97910
// Address: 0x1b97910
//
__int64 __fastcall sub_1B97910(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r13
  __int64 v6; // rdi
  _BOOL4 v7; // r8d
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rax
  int v11; // esi
  unsigned int v12; // ecx
  int *v13; // rdx
  int v14; // r8d
  int v15; // eax
  __int64 v16; // rsi
  int v17; // edx
  int v18; // edi
  unsigned int v19; // eax
  __int64 v20; // rcx
  int v21; // edx
  int v22; // r10d
  int v23; // [rsp+4h] [rbp-2Ch] BYREF
  __int64 v24[5]; // [rsp+8h] [rbp-28h] BYREF

  result = *(unsigned int *)(a1 + 88);
  v3 = *(_QWORD *)(a1 + 456);
  v23 = result;
  if ( (_DWORD)result != 1 )
  {
    v6 = (unsigned __int8)sub_1B97860(v3 + 200, &v23, v24)
       ? v24[0]
       : *(_QWORD *)(v3 + 208) + 80LL * *(unsigned int *)(v3 + 224);
    v7 = sub_13A0E30(v6 + 8, a2);
    result = 1;
    if ( !v7 )
    {
      v8 = *(_QWORD *)(a1 + 456);
      v9 = *(_QWORD *)(v8 + 144);
      v10 = *(unsigned int *)(v8 + 160);
      if ( (_DWORD)v10 )
      {
        v11 = *(_DWORD *)(a1 + 88);
        v12 = (v10 - 1) & (37 * v11);
        v13 = (int *)(v9 + 40LL * v12);
        v14 = *v13;
        if ( v11 == *v13 )
          goto LABEL_7;
        v21 = 1;
        while ( v14 != -1 )
        {
          v22 = v21 + 1;
          v12 = (v10 - 1) & (v21 + v12);
          v13 = (int *)(v9 + 40LL * v12);
          v14 = *v13;
          if ( v11 == *v13 )
            goto LABEL_7;
          v21 = v22;
        }
      }
      v13 = (int *)(v9 + 40 * v10);
LABEL_7:
      v15 = v13[8];
      if ( v15 )
      {
        v16 = *((_QWORD *)v13 + 2);
        v17 = v15 - 1;
        v18 = 1;
        v19 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v20 = *(_QWORD *)(v16 + 16LL * v19);
        if ( a2 == v20 )
          return 1;
        while ( v20 != -8 )
        {
          v19 = v17 & (v18 + v19);
          v20 = *(_QWORD *)(v16 + 16LL * v19);
          if ( a2 == v20 )
            return 1;
          ++v18;
        }
      }
      return 0;
    }
  }
  return result;
}
