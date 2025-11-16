// Function: sub_A69870
// Address: 0xa69870
//
__int64 (__fastcall *__fastcall sub_A69870(__int64 a1, _BYTE *a2, char a3))(_QWORD *, _QWORD *, __int64)
{
  __int64 v4; // rax
  __int64 v5; // rax
  char v6; // dl
  __int64 v8; // rdx
  __int64 *v9; // rax
  __int64 *v10; // rcx
  __int64 v11; // rdx
  _QWORD v12[18]; // [rsp+0h] [rbp-90h] BYREF

  if ( *(_BYTE *)a1 == 85 )
  {
    v4 = *(_QWORD *)(a1 - 32);
    if ( v4 )
    {
      if ( !*(_BYTE *)v4 && *(_QWORD *)(v4 + 24) == *(_QWORD *)(a1 + 80) && (*(_BYTE *)(v4 + 33) & 0x20) != 0 )
      {
        v8 = 4LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
        if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
        {
          v9 = *(__int64 **)(a1 - 8);
          v10 = &v9[v8];
        }
        else
        {
          v10 = (__int64 *)a1;
          v9 = (__int64 *)(a1 - v8 * 8);
        }
        for ( ; v10 != v9; v9 += 4 )
        {
          v11 = *v9;
          if ( *v9 && *(_BYTE *)v11 == 24 && (unsigned __int8)(**(_BYTE **)(v11 + 24) - 5) <= 0x1Fu )
            break;
        }
      }
    }
  }
  v5 = sub_A4F760((unsigned __int8 *)a1);
  sub_A558A0((__int64)v12, v5, v6);
  sub_A693B0(a1, a2, (__int64)v12, a3);
  return sub_A55520(v12, (__int64)a2);
}
