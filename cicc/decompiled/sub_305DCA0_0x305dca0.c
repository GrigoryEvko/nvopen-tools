// Function: sub_305DCA0
// Address: 0x305dca0
//
__int64 __fastcall sub_305DCA0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 (*v4)(void); // rax
  unsigned int v5; // r13d
  __int64 v7; // rdx
  __int64 v8; // rdx
  _QWORD v9[2]; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v10[4]; // [rsp+10h] [rbp-20h] BYREF

  v4 = *(__int64 (**)(void))(**(_QWORD **)(a1 + 32) + 1376LL);
  if ( (char *)v4 != (char *)sub_302EF50 )
    return v4();
  v5 = 0;
  if ( *(_BYTE *)(a2 + 8) == 12 && *(_BYTE *)(a3 + 8) == 12 )
  {
    v9[0] = sub_BCAE30(a2);
    v9[1] = v7;
    if ( sub_CA1930(v9) == 64 )
    {
      v10[0] = sub_BCAE30(a3);
      v10[1] = v8;
      LOBYTE(v5) = sub_CA1930(v10) == 32;
    }
  }
  return v5;
}
