// Function: sub_E213F0
// Address: 0xe213f0
//
size_t __fastcall sub_E213F0(__int64 a1, size_t a2, const void *a3)
{
  _QWORD *v4; // rdx
  __int64 v5; // rax
  void *v6; // r8
  __int64 v7; // r14
  __int64 *v8; // rax
  __int64 *v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rax

  v4 = *(_QWORD **)(a1 + 16);
  v5 = v4[1];
  v6 = (void *)(v5 + *v4);
  v4[1] = a2 + v5;
  if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
  {
    v7 = 4096;
    if ( a2 >= 0x1000 )
      v7 = a2;
    v8 = (__int64 *)sub_22077B0(32);
    v9 = v8;
    if ( v8 )
    {
      *v8 = 0;
      v8[1] = 0;
      v8[2] = 0;
      v8[3] = 0;
    }
    v10 = sub_2207820(v7);
    v9[2] = v7;
    *v9 = v10;
    v6 = (void *)v10;
    v11 = *(_QWORD *)(a1 + 16);
    v9[1] = a2;
    v9[3] = v11;
    *(_QWORD *)(a1 + 16) = v9;
  }
  if ( a2 )
    memcpy(v6, a3, a2);
  return a2;
}
