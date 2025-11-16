// Function: sub_B71A20
// Address: 0xb71a20
//
__int64 __fastcall sub_B71A20(__int64 a1, const void *a2, size_t a3)
{
  unsigned int v4; // eax
  __int64 v5; // rdx
  __int64 v6; // rax
  unsigned int v7; // r14d
  __int64 v8; // r8
  __int64 v10; // rax
  __int64 *v11; // r10
  __int64 v12; // rcx
  __int64 *v13; // rax
  __int64 *v14; // rax
  __int64 v15; // [rsp+8h] [rbp-68h]
  __int64 *v16; // [rsp+10h] [rbp-60h]
  int v17; // [rsp+1Ch] [rbp-54h]

  v17 = *(_DWORD *)(a1 + 3428);
  v4 = sub_C92610(a2, a3);
  v5 = (unsigned int)sub_C92740(a1 + 3416, a2, a3, v4);
  v6 = *(_QWORD *)(a1 + 3416);
  v7 = v5;
  v8 = *(_QWORD *)(v6 + 8 * v5);
  if ( v8 )
  {
    if ( v8 != -8 )
      return v8;
    --*(_DWORD *)(a1 + 3432);
  }
  v16 = (__int64 *)(v6 + 8 * v5);
  v10 = sub_C7D670(a3 + 17, 8);
  v11 = v16;
  v12 = v10;
  if ( a3 )
  {
    v15 = v10;
    memcpy((void *)(v10 + 16), a2, a3);
    v11 = v16;
    v12 = v15;
  }
  *(_BYTE *)(v12 + a3 + 16) = 0;
  *(_QWORD *)v12 = a3;
  *(_DWORD *)(v12 + 8) = v17;
  *v11 = v12;
  ++*(_DWORD *)(a1 + 3428);
  v13 = (__int64 *)(*(_QWORD *)(a1 + 3416) + 8LL * (unsigned int)sub_C929D0(a1 + 3416, v7));
  v8 = *v13;
  if ( !*v13 || v8 == -8 )
  {
    v14 = v13 + 1;
    do
    {
      do
        v8 = *v14++;
      while ( !v8 );
    }
    while ( v8 == -8 );
  }
  return v8;
}
