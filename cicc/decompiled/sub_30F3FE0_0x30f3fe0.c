// Function: sub_30F3FE0
// Address: 0x30f3fe0
//
__int64 __fastcall sub_30F3FE0(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE *v7; // rsi
  __int64 v8; // rdi
  __int64 v9; // r15
  __int64 v10; // r12
  __int64 v11; // rbx
  __int64 v12; // rax
  _QWORD *v13; // rax
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rdx
  _QWORD *v18; // [rsp+0h] [rbp-80h]
  unsigned __int8 v19; // [rsp+27h] [rbp-59h]
  __int64 v20; // [rsp+28h] [rbp-58h]
  unsigned __int64 v21[2]; // [rsp+30h] [rbp-50h] BYREF
  _BYTE v22[64]; // [rsp+40h] [rbp-40h] BYREF

  v7 = *(_BYTE **)(a1 + 8);
  v8 = *(_QWORD *)(a1 + 104);
  v21[0] = (unsigned __int64)v22;
  v21[1] = 0x400000000LL;
  v19 = sub_30B8930(v8, v7, a2, a3, (__int64)v21);
  if ( v19 )
  {
    v9 = 1;
    v20 = *(unsigned int *)(a3 + 8);
    if ( v20 != 1 )
    {
      do
      {
        v10 = *(_QWORD *)(a1 + 104);
        v11 = *(int *)(v21[0] + 4LL * (unsigned int)(v9 - 1));
        v12 = sub_D95540(*(_QWORD *)(*(_QWORD *)a3 + 8LL * (unsigned int)v9));
        v13 = sub_DA2C50(v10, v12, v11, 0);
        v16 = *(unsigned int *)(a1 + 72);
        if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 76) )
        {
          v18 = v13;
          sub_C8D5F0(a1 + 64, (const void *)(a1 + 80), v16 + 1, 8u, v14, v15);
          v16 = *(unsigned int *)(a1 + 72);
          v13 = v18;
        }
        ++v9;
        *(_QWORD *)(*(_QWORD *)(a1 + 64) + 8 * v16) = v13;
        ++*(_DWORD *)(a1 + 72);
      }
      while ( v20 != v9 );
    }
  }
  if ( (_BYTE *)v21[0] != v22 )
    _libc_free(v21[0]);
  return v19;
}
