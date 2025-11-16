// Function: sub_2A73B10
// Address: 0x2a73b10
//
__int64 __fastcall sub_2A73B10(__int64 a1, __int64 a2)
{
  bool v2; // dl
  unsigned int v3; // r8d
  __int64 v4; // rdi
  unsigned int v5; // ecx
  __int64 v6; // rbx
  __int64 v7; // rax
  int v9; // r10d
  _QWORD v10[2]; // [rsp+0h] [rbp-30h] BYREF
  __int64 v11; // [rsp+10h] [rbp-20h]

  v10[0] = 0;
  v10[1] = 0;
  v11 = a2;
  v2 = a2 != 0 && a2 != -4096 && a2 != -8192;
  if ( v2 )
  {
    sub_BD73F0((__int64)v10);
    a2 = v11;
    v2 = v11 != -4096 && v11 != 0 && v11 != -8192;
  }
  v3 = *(_DWORD *)(a1 + 280);
  v4 = *(_QWORD *)(a1 + 264);
  if ( v3 )
  {
    v5 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = v4 + 32LL * v5;
    v7 = *(_QWORD *)(v6 + 16);
    if ( a2 == v7 )
      goto LABEL_5;
    v9 = 1;
    while ( v7 != -4096 )
    {
      v5 = (v3 - 1) & (v9 + v5);
      v6 = v4 + 32LL * v5;
      v7 = *(_QWORD *)(v6 + 16);
      if ( v7 == a2 )
        goto LABEL_5;
      ++v9;
    }
  }
  v6 = v4 + 32LL * v3;
LABEL_5:
  if ( v2 )
    sub_BD60C0(v10);
  return *(unsigned int *)(v6 + 24);
}
