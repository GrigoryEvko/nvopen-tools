// Function: sub_3249CA0
// Address: 0x3249ca0
//
void __fastcall sub_3249CA0(__int64 *a1, __int64 a2, unsigned int a3, __int64 a4)
{
  unsigned int v5; // eax
  int v6; // [rsp-2Ch] [rbp-2Ch]

  if ( a3 )
  {
    v5 = (*(__int64 (__fastcall **)(__int64 *, __int64))(*a1 + 80))(a1, a4);
    BYTE2(v6) = 0;
    sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), 58, v6, v5);
    BYTE2(v6) = 0;
    sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), 59, v6, a3);
  }
}
