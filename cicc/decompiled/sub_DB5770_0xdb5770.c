// Function: sub_DB5770
// Address: 0xdb5770
//
__int64 __fastcall sub_DB5770(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v5; // r14d
  __int64 v6; // rbx
  __int64 v7; // r15
  __int64 v8; // rbx
  unsigned int v9; // eax
  const void **v11; // [rsp+0h] [rbp-60h]
  unsigned int v13; // [rsp+14h] [rbp-4Ch]
  const void *v14; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-38h]

  v5 = *(_DWORD *)(a3 + 32);
  v6 = *(_QWORD *)(a4 + 40);
  v11 = (const void **)(a3 + 24);
  v13 = v5;
  if ( (unsigned int)v6 <= 1 )
  {
    if ( v5 )
    {
LABEL_16:
      *(_DWORD *)(a1 + 8) = v13;
      if ( v13 > 0x40 )
        sub_C43780(a1, v11);
      else
        *(_QWORD *)a1 = *(_QWORD *)(a3 + 24);
    }
    else
    {
      *(_DWORD *)(a1 + 8) = 0;
LABEL_8:
      *(_QWORD *)a1 = 0;
    }
  }
  else
  {
    v7 = 8;
    v8 = 8LL * (unsigned int)v6;
    do
    {
      if ( !v5 )
        goto LABEL_7;
      v9 = sub_DB55F0(a2, *(_QWORD *)(*(_QWORD *)(a4 + 32) + v7));
      if ( v5 > v9 )
        v5 = v9;
      v7 += 8;
    }
    while ( v8 != v7 );
    if ( !v5 )
    {
LABEL_7:
      *(_DWORD *)(a1 + 8) = v13;
      if ( v13 <= 0x40 )
        goto LABEL_8;
      sub_C43690(a1, 0, 0);
      return a1;
    }
    if ( v13 <= v5 )
    {
      v13 = *(_DWORD *)(a3 + 32);
      goto LABEL_16;
    }
    sub_C44740((__int64)&v14, (char **)v11, v5);
    sub_C449B0(a1, &v14, v13);
    if ( v15 > 0x40 && v14 )
      j_j___libc_free_0_0(v14);
  }
  return a1;
}
