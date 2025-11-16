// Function: sub_11D9940
// Address: 0x11d9940
//
__int64 __fastcall sub_11D9940(__int64 a1, int a2, char a3)
{
  __int64 v3; // rbp
  int v5; // ecx
  unsigned int v6; // edx
  __int64 result; // rax
  void *v8; // [rsp-18h] [rbp-18h] BYREF
  int v9; // [rsp-10h] [rbp-10h]
  char v10; // [rsp-Ch] [rbp-Ch]
  __int64 v11; // [rsp-8h] [rbp-8h]

  v5 = *(_DWORD *)(a1 + 152);
  v6 = *(_DWORD *)(a1 + 136);
  result = *(unsigned __int8 *)(a1 + 156);
  if ( a3 || !(_BYTE)result || v6 != v5 )
  {
    v11 = v3;
    v10 = result;
    v9 = v5;
    v8 = &unk_49D9728;
    return sub_C56860(a1 + 160, a1, v6, (__int64)&v8, a2);
  }
  return result;
}
