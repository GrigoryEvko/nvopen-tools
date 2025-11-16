// Function: sub_30CC6B0
// Address: 0x30cc6b0
//
__int64 __fastcall sub_30CC6B0(__int64 a1, __int64 *a2, __int64 a3, char a4)
{
  __int64 v6; // rdx
  __int64 v7; // rax
  _BOOL8 v8; // rcx

  if ( a4 )
  {
    v6 = sub_B491C0(a3);
    v7 = *(_QWORD *)(a3 - 32);
    if ( v7 )
    {
      if ( *(_BYTE *)v7 )
      {
        v7 = 0;
      }
      else if ( *(_QWORD *)(v7 + 24) != *(_QWORD *)(a3 + 80) )
      {
        v7 = 0;
      }
    }
    v8 = 0;
    if ( v6 != v7 )
    {
      sub_30CC5F0((__int64)a2, a3);
      v8 = (unsigned int)sub_30CC550(a3, a2[2]) == 1;
    }
    (*(void (__fastcall **)(__int64, __int64 *, __int64, _BOOL8))(*a2 + 56))(a1, a2, a3, v8);
    return a1;
  }
  else
  {
    (*(void (**)(void))(*a2 + 48))();
    return a1;
  }
}
