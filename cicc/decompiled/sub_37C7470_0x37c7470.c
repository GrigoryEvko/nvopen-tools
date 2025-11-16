// Function: sub_37C7470
// Address: 0x37c7470
//
char __fastcall sub_37C7470(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  unsigned __int64 v5; // rdx
  unsigned __int64 v6; // rax
  __int64 v7; // rdi
  int v8; // edx
  __int64 (*v9)(); // rcx
  unsigned __int64 v11[5]; // [rsp+8h] [rbp-28h] BYREF

  v5 = sub_37C7360(a1, a2);
  v6 = HIDWORD(v5);
  v11[0] = v5;
  if ( BYTE4(v5) )
  {
    v7 = *(_QWORD *)(a1 + 32);
    v8 = 0;
    v9 = *(__int64 (**)())(*(_QWORD *)v7 + 136LL);
    LOBYTE(v6) = 0;
    if ( v9 != sub_2E85450 )
    {
      v8 = ((__int64 (__fastcall *)(__int64, __int64, unsigned __int64 *))v9)(v7, a2, v11);
      LOBYTE(v6) = v8 != 0;
    }
    *a4 = v8;
  }
  return v6;
}
