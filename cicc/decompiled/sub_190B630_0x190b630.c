// Function: sub_190B630
// Address: 0x190b630
//
__int64 __fastcall sub_190B630(__int64 a1)
{
  unsigned int v1; // eax
  __int64 v3; // rdx
  __int64 v4; // rdi
  __int64 v5; // rsi
  __int64 v6; // rax
  _QWORD v8[2]; // [rsp-38h] [rbp-38h] BYREF
  int v9; // [rsp-28h] [rbp-28h]

  v1 = *(_DWORD *)(a1 + 800);
  if ( !v1 )
    return 0;
  do
  {
    v3 = *(_QWORD *)(a1 + 792) + 16LL * v1 - 16;
    v4 = *(_QWORD *)v3;
    v5 = *(unsigned int *)(v3 + 8);
    *(_DWORD *)(a1 + 800) = v1 - 1;
    v6 = *(_QWORD *)(a1 + 24);
    v8[1] = 0;
    v9 = (int)&loc_1000000;
    v8[0] = v6;
    sub_1AAC5F0(v4, v5, v8);
    v1 = *(_DWORD *)(a1 + 800);
  }
  while ( v1 );
  if ( *(_QWORD *)a1 )
    sub_1413520(*(_QWORD *)a1);
  sub_190B570(a1);
  *(_BYTE *)(a1 + 784) = 1;
  return 1;
}
