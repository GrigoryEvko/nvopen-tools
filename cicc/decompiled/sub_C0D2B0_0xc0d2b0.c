// Function: sub_C0D2B0
// Address: 0xc0d2b0
//
__int64 __fastcall sub_C0D2B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // r12
  void (__fastcall *v8)(__int64, __int64); // rax
  __int64 (__fastcall *v10)(__int64, __int64, __int64, __int64, __int64); // rax
  __int64 (__fastcall *v11)(__int64, __int64, __int64, __int64); // rax

  if ( *(_DWORD *)(a2 + 52) > 8u )
  {
    v7 = 0;
  }
  else
  {
    switch ( *(_DWORD *)(a2 + 52) )
    {
      case 0:
        BUG();
      case 1:
        v7 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(a1 + 152))(a3, a4, a5, a6);
        break;
      case 2:
        v7 = sub_E7D5E0(a3, a4, a5, a6);
        break;
      case 3:
        v10 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(a1 + 168);
        if ( v10 )
          v7 = v10(a2, a3, a4, a5, a6);
        else
          v7 = sub_E7EE90(a3, a4, a5, a6);
        break;
      case 4:
        v7 = sub_E81CF0(a3, a4, a5, a6);
        break;
      case 5:
        v11 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(a1 + 160);
        if ( v11 )
          v7 = v11(a3, a4, a5, a6);
        else
          v7 = sub_E834C0(a3, a4, a5, a6, 0, 0);
        break;
      case 6:
        v7 = sub_E97400(a3, a4, a5, a6);
        break;
      case 7:
        v7 = sub_EA1E70(a3, a4, a5, a6);
        break;
      case 8:
        v7 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(a1 + 176))(a2, a3, a4, a5, a6);
        break;
    }
  }
  v8 = *(void (__fastcall **)(__int64, __int64))(a1 + 200);
  if ( v8 )
    v8(v7, a7);
  return v7;
}
