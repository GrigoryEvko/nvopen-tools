// Function: sub_352B3E0
// Address: 0x352b3e0
//
unsigned __int64 __fastcall sub_352B3E0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rbx
  __int64 v4; // rax
  const char *v5; // rax
  unsigned __int64 result; // rax
  _QWORD v7[2]; // [rsp+0h] [rbp-70h] BYREF
  __int64 (__fastcall *v8)(_QWORD *, _QWORD *, __int64); // [rsp+10h] [rbp-60h]
  const char *v9; // [rsp+20h] [rbp-50h] BYREF
  char v10; // [rsp+40h] [rbp-30h]
  char v11; // [rsp+41h] [rbp-2Fh]

  v2 = *(_QWORD *)(a2 + 32);
  v3 = v2 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF);
  v4 = v2 + 40LL * (unsigned int)sub_2E88F80(a2);
  if ( v3 == v4 )
  {
LABEL_8:
    result = sub_2EBEE90(*(_QWORD *)(*(_QWORD *)(a1 + 144) + 32LL), *(_DWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
    if ( result )
      return result;
    sub_2EE7320(v7, a1 + 144, a2);
    v11 = 1;
    v5 = "Convergence control tokens must have unique definitions.";
  }
  else
  {
    while ( (*(_BYTE *)(v4 + 3) & 0x10) == 0 )
    {
      v4 += 40;
      if ( v3 == v4 )
        goto LABEL_8;
    }
    sub_2EE7320(v7, a1 + 144, a2);
    v11 = 1;
    v5 = "Convergence control tokens are defined explicitly.";
  }
  v9 = v5;
  v10 = 3;
  sub_352B2E0((_BYTE *)a1, (__int64)&v9, v7, 1);
  result = (unsigned __int64)v8;
  if ( v8 )
    return v8(v7, v7, 3);
  return result;
}
