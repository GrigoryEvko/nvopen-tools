// Function: sub_39062D0
// Address: 0x39062d0
//
__int64 __fastcall sub_39062D0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // rax
  unsigned int v6; // edx
  _QWORD *v7; // rdi
  __int64 result; // rax
  __int64 v9; // rdi
  __int64 v10; // [rsp+18h] [rbp-38h] BYREF
  const char *v11; // [rsp+20h] [rbp-30h] BYREF
  char v12; // [rsp+30h] [rbp-20h]
  char v13; // [rsp+31h] [rbp-1Fh]

  v2 = *(_QWORD *)(a1 + 8);
  v10 = 0;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 40LL))(v2) + 8) == 9
    || (result = sub_3909510(*(_QWORD *)(a1 + 8), &v10), !(_BYTE)result) )
  {
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
    {
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
      v5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
      v6 = *(_DWORD *)(v5 + 120);
      v7 = (_QWORD *)v5;
      result = 0;
      if ( v6 )
      {
        (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*v7 + 160LL))(v7, *(_QWORD *)(v7[14] + 32LL * v6 - 32), v10);
        return 0;
      }
    }
    else
    {
      v9 = *(_QWORD *)(a1 + 8);
      v13 = 1;
      v11 = "unexpected token in directive";
      v12 = 3;
      return sub_3909CF0(v9, &v11, 0, 0, v3, v4);
    }
  }
  return result;
}
