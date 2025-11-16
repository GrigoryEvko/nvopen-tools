// Function: sub_90ADD0
// Address: 0x90add0
//
__int64 *__fastcall sub_90ADD0(__int64 a1, const void *a2, size_t a3)
{
  unsigned int v4; // eax
  unsigned int v5; // r8d
  __int64 *v6; // r12
  __int64 v7; // rax
  unsigned int v8; // r8d
  __int64 v9; // r15
  __int64 v10; // rax
  unsigned int v12; // [rsp+Ch] [rbp-34h]

  v4 = sub_C92610(a2, a3);
  v5 = sub_C92740(a1, a2, a3, v4);
  v6 = (__int64 *)(*(_QWORD *)a1 + 8LL * v5);
  if ( *v6 )
  {
    if ( *v6 != -8 )
      return v6;
    --*(_DWORD *)(a1 + 16);
  }
  v12 = v5;
  v7 = sub_C7D670(a3 + 17, 8);
  v8 = v12;
  v9 = v7;
  if ( a3 )
  {
    memcpy((void *)(v7 + 16), a2, a3);
    v8 = v12;
  }
  *(_BYTE *)(v9 + a3 + 16) = 0;
  *(_QWORD *)v9 = a3;
  *(_DWORD *)(v9 + 8) = 0;
  *v6 = v9;
  ++*(_DWORD *)(a1 + 12);
  v6 = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_C929D0(a1, v8));
  v10 = *v6;
  if ( *v6 )
    goto LABEL_8;
  do
  {
    do
    {
      v10 = v6[1];
      ++v6;
    }
    while ( !v10 );
LABEL_8:
    ;
  }
  while ( v10 == -8 );
  return v6;
}
