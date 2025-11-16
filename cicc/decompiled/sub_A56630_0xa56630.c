// Function: sub_A56630
// Address: 0xa56630
//
__int64 __fastcall sub_A56630(__int64 a1, const void *a2, size_t a3)
{
  int v4; // r13d
  unsigned int v5; // eax
  unsigned int v6; // r8d
  __int64 *v7; // r9
  __int64 result; // rax
  __int64 v9; // rax
  unsigned int v10; // r8d
  __int64 *v11; // r9
  __int64 v12; // rcx
  __int64 *v13; // rdx
  __int64 v14; // [rsp+8h] [rbp-48h]
  __int64 *v15; // [rsp+10h] [rbp-40h]
  unsigned int v16; // [rsp+1Ch] [rbp-34h]

  v4 = *(_DWORD *)(a1 + 392);
  *(_DWORD *)(a1 + 392) = v4 + 1;
  v5 = sub_C92610(a2, a3);
  v6 = sub_C92740(a1 + 368, a2, a3, v5);
  v7 = (__int64 *)(*(_QWORD *)(a1 + 368) + 8LL * v6);
  result = *v7;
  if ( *v7 )
  {
    if ( result != -8 )
      goto LABEL_3;
    --*(_DWORD *)(a1 + 384);
  }
  v15 = v7;
  v16 = v6;
  v9 = sub_C7D670(a3 + 17, 8);
  v10 = v16;
  v11 = v15;
  v12 = v9;
  if ( a3 )
  {
    v14 = v9;
    memcpy((void *)(v9 + 16), a2, a3);
    v10 = v16;
    v11 = v15;
    v12 = v14;
  }
  *(_BYTE *)(v12 + a3 + 16) = 0;
  *(_QWORD *)v12 = a3;
  *(_DWORD *)(v12 + 8) = 0;
  *v11 = v12;
  ++*(_DWORD *)(a1 + 380);
  v13 = (__int64 *)(*(_QWORD *)(a1 + 368) + 8LL * (unsigned int)sub_C929D0(a1 + 368, v10));
  result = *v13;
  if ( *v13 )
    goto LABEL_9;
  do
  {
    do
    {
      result = v13[1];
      ++v13;
    }
    while ( !result );
LABEL_9:
    ;
  }
  while ( result == -8 );
LABEL_3:
  *(_DWORD *)(result + 8) = v4;
  return result;
}
