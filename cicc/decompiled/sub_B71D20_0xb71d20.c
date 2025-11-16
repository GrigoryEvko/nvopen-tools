// Function: sub_B71D20
// Address: 0xb71d20
//
__int64 __fastcall sub_B71D20(__int64 a1, const void *a2, size_t a3)
{
  unsigned int v4; // eax
  unsigned int v5; // r14d
  __int64 *v6; // r9
  __int64 v7; // rdx
  __int64 v9; // rax
  __int64 *v10; // r9
  __int64 v11; // rcx
  __int64 *v12; // rax
  __int64 *v13; // rax
  __int64 v14; // [rsp+8h] [rbp-68h]
  __int64 *v15; // [rsp+10h] [rbp-60h]
  int v16; // [rsp+1Ch] [rbp-54h]

  v16 = *(_DWORD *)(a1 + 3452);
  v4 = sub_C92610(a2, a3);
  v5 = sub_C92740(a1 + 3440, a2, a3, v4);
  v6 = (__int64 *)(*(_QWORD *)(a1 + 3440) + 8LL * v5);
  v7 = *v6;
  if ( *v6 )
  {
    if ( v7 != -8 )
      return *(unsigned __int8 *)(v7 + 8);
    --*(_DWORD *)(a1 + 3456);
  }
  v15 = v6;
  v9 = sub_C7D670(a3 + 17, 8);
  v10 = v15;
  v11 = v9;
  if ( a3 )
  {
    v14 = v9;
    memcpy((void *)(v9 + 16), a2, a3);
    v10 = v15;
    v11 = v14;
  }
  *(_BYTE *)(v11 + a3 + 16) = 0;
  *(_QWORD *)v11 = a3;
  *(_BYTE *)(v11 + 8) = v16;
  *v10 = v11;
  ++*(_DWORD *)(a1 + 3452);
  v12 = (__int64 *)(*(_QWORD *)(a1 + 3440) + 8LL * (unsigned int)sub_C929D0(a1 + 3440, v5));
  v7 = *v12;
  if ( !*v12 || v7 == -8 )
  {
    v13 = v12 + 1;
    do
    {
      do
        v7 = *v13++;
      while ( !v7 );
    }
    while ( v7 == -8 );
  }
  return *(unsigned __int8 *)(v7 + 8);
}
