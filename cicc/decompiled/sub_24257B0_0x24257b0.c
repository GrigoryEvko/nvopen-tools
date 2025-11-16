// Function: sub_24257B0
// Address: 0x24257b0
//
__int64 __fastcall sub_24257B0(__int64 *a1, _BYTE *a2, size_t a3)
{
  int v4; // eax
  __int64 v5; // rax
  _QWORD *v7; // r15
  __int64 v8; // rax
  __int64 *v9; // rdx
  __int64 *v10; // [rsp+0h] [rbp-40h]
  unsigned int v11; // [rsp+Ch] [rbp-34h]

  v4 = sub_C92610();
  v11 = sub_C92740((__int64)(a1 + 12), a2, a3, v4);
  v10 = (__int64 *)(a1[12] + 8LL * v11);
  v5 = *v10;
  if ( *v10 )
  {
    if ( v5 != -8 )
      return v5 + 8;
    --*((_DWORD *)a1 + 28);
  }
  v7 = (_QWORD *)sub_C7D670(a3 + 193, 8);
  if ( a3 )
    memcpy(v7 + 24, a2, a3);
  v8 = *a1;
  *((_BYTE *)v7 + a3 + 192) = 0;
  v7[1] = v8;
  v7[2] = v7 + 4;
  *v7 = a3;
  sub_2425560(v7 + 2, a2, (__int64)&a2[a3]);
  v7[6] = v7 + 8;
  v7[7] = 0x2000000000LL;
  *v10 = (__int64)v7;
  ++*((_DWORD *)a1 + 27);
  v9 = (__int64 *)(a1[12] + 8LL * (unsigned int)sub_C929D0(a1 + 12, v11));
  v5 = *v9;
  if ( *v9 )
    goto LABEL_9;
  do
  {
    do
    {
      v5 = v9[1];
      ++v9;
    }
    while ( !v5 );
LABEL_9:
    ;
  }
  while ( v5 == -8 );
  return v5 + 8;
}
