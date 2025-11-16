// Function: sub_D19730
// Address: 0xd19730
//
__int64 __fastcall sub_D19730(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 v9; // rcx
  unsigned int v10; // edx
  __int64 *v11; // rsi
  __int64 v12; // rdi
  unsigned int v13; // eax
  int v15; // esi
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rdx
  unsigned int v19; // eax
  unsigned __int64 v20; // rdx
  int v21; // r9d
  _QWORD v22[6]; // [rsp+0h] [rbp-30h] BYREF

  sub_D19710(a2, a2, a3, a4, a5, a6);
  v8 = *(unsigned int *)(a2 + 344);
  v9 = *(_QWORD *)(a2 + 328);
  if ( (_DWORD)v8 )
  {
    v10 = (v8 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v11 = (__int64 *)(v9 + 24LL * v10);
    v12 = *v11;
    if ( a3 == *v11 )
    {
LABEL_3:
      if ( v11 != (__int64 *)(v9 + 24 * v8) )
      {
        v13 = *((_DWORD *)v11 + 4);
        *(_DWORD *)(a1 + 8) = v13;
        if ( v13 <= 0x40 )
        {
          *(_QWORD *)a1 = v11[1];
          return a1;
        }
        sub_C43780(a1, (const void **)v11 + 1);
        return a1;
      }
    }
    else
    {
      v15 = 1;
      while ( v12 != -4096 )
      {
        v21 = v15 + 1;
        v10 = (v8 - 1) & (v15 + v10);
        v11 = (__int64 *)(v9 + 24LL * v10);
        v12 = *v11;
        if ( a3 == *v11 )
          goto LABEL_3;
        v15 = v21;
      }
    }
  }
  v16 = sub_B43CC0(a3);
  v17 = *(_QWORD *)(a3 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v17 + 8) - 17 <= 1 )
    v17 = **(_QWORD **)(v17 + 16);
  v22[0] = sub_9208B0(v16, v17);
  v22[1] = v18;
  v19 = sub_CA1930(v22);
  *(_DWORD *)(a1 + 8) = v19;
  if ( v19 > 0x40 )
  {
    sub_C43690(a1, -1, 1);
    return a1;
  }
  v20 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v19;
  if ( !v19 )
    v20 = 0;
  *(_QWORD *)a1 = v20;
  return a1;
}
