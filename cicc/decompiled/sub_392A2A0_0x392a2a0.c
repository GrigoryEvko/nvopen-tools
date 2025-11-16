// Function: sub_392A2A0
// Address: 0x392a2a0
//
__int64 __fastcall sub_392A2A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v6; // r15d
  const void *v7; // rax
  unsigned int v9; // eax
  const void *v10; // rax
  const void *v11; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v12; // [rsp+8h] [rbp-38h]

  v6 = *(_DWORD *)(a4 + 8);
  if ( v6 > 0x40 )
  {
    if ( v6 - (unsigned int)sub_16A57B0(a4) > 0x40 )
    {
      v12 = v6;
      sub_16A4FD0((__int64)&v11, (const void **)a4);
      v9 = v12;
      *(_QWORD *)(a1 + 8) = a2;
      *(_DWORD *)a1 = 5;
      *(_DWORD *)(a1 + 32) = v9;
      v10 = v11;
      *(_QWORD *)(a1 + 16) = a3;
      *(_QWORD *)(a1 + 24) = v10;
      return a1;
    }
    v12 = v6;
    sub_16A4FD0((__int64)&v11, (const void **)a4);
    v6 = v12;
    v7 = v11;
  }
  else
  {
    v7 = *(const void **)a4;
  }
  *(_DWORD *)a1 = 4;
  *(_QWORD *)(a1 + 8) = a2;
  *(_QWORD *)(a1 + 16) = a3;
  *(_DWORD *)(a1 + 32) = v6;
  *(_QWORD *)(a1 + 24) = v7;
  return a1;
}
