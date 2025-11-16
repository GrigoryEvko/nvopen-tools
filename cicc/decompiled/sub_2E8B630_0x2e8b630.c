// Function: sub_2E8B630
// Address: 0x2e8b630
//
char __fastcall sub_2E8B630(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5)
{
  __int64 v5; // rbp
  __int64 v6; // rcx
  __int64 v7; // rdx
  char result; // al
  char v9; // [rsp-9h] [rbp-9h] BYREF
  __int64 v10; // [rsp-8h] [rbp-8h]

  v6 = *(unsigned __int16 *)(a1 + 68);
  v7 = (unsigned __int16)v6;
  if ( (unsigned __int16)v6 > 0x2Bu
    || (result = ((0x80200C00000uLL >> v6) & 1) == 0, ((0x80200C00000uLL >> v6) & 1) == 0) )
  {
    v10 = v5;
    v9 = 0;
    LOBYTE(v7) = (unsigned __int16)v6 == 0;
    result = v7 | ((unsigned __int16)v6 == 68);
    if ( !result )
      return sub_2E8B400(a1, (__int64)&v9, v7, v6, a5);
  }
  return result;
}
