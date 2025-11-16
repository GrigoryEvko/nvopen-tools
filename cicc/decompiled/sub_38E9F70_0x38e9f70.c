// Function: sub_38E9F70
// Address: 0x38e9f70
//
__int64 __fastcall sub_38E9F70(__int64 a1)
{
  __int64 v1; // rbx
  __int64 result; // rax
  __int64 *v3; // r10
  __int64 v4; // rsi
  __int64 v5; // rdi
  __int64 v6; // rdx
  unsigned __int8 v7; // [rsp+Fh] [rbp-41h]
  _DWORD v8[4]; // [rsp+10h] [rbp-40h] BYREF
  _BYTE v9[48]; // [rsp+20h] [rbp-30h] BYREF

  v1 = *(_QWORD *)(a1 + 320);
  result = *(unsigned __int8 *)(v1 + 1041);
  if ( (_BYTE)result )
  {
    if ( !*(_DWORD *)(v1 + 1044) )
    {
      v3 = *(__int64 **)(a1 + 328);
      v7 = *(_BYTE *)(v1 + 1041);
      v4 = *(_QWORD *)(v1 + 944);
      v5 = *(_QWORD *)(v1 + 952);
      v6 = *v3;
      v9[16] = 0;
      (*(void (__fastcall **)(_DWORD *, __int64 *, _QWORD, _QWORD, _QWORD, _QWORD, __int64, __int64, _BYTE *, _QWORD))(v6 + 568))(
        v8,
        v3,
        0,
        0,
        0,
        0,
        v4,
        v5,
        v9,
        0);
      *(_DWORD *)(v1 + 1044) = v8[0];
      return v7;
    }
  }
  return result;
}
