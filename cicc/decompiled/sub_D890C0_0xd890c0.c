// Function: sub_D890C0
// Address: 0xd890c0
//
__int64 __fastcall sub_D890C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v6; // eax
  unsigned int v7; // eax
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v14[6]; // [rsp+10h] [rbp-30h] BYREF

  if ( (unsigned __int8)sub_D97040(*(_QWORD *)(a2 + 16), *(_QWORD *)(a3 + 8))
    && (unsigned __int8)sub_D97040(*(_QWORD *)(a2 + 16), *(_QWORD *)(a4 + 8)) )
  {
    v9 = sub_D85880(a2, a3);
    v10 = sub_D85880(a2, a4);
    if ( v9 )
    {
      if ( v10 )
      {
        v11 = sub_DCC810(*(_QWORD *)(a2 + 16), v9, v10, 0, 0);
        if ( !(unsigned __int8)sub_D96A50(v11) )
        {
          v12 = sub_DBB9F0(*(_QWORD *)(a2 + 16), v11, 1, 0);
          sub_AAF450((__int64)v13, v12);
          if ( sub_D85770((__int64)v13) )
            sub_AAF450(a1, a2 + 32);
          else
            sub_AB4E00(a1, (__int64)v13, *(_DWORD *)(a2 + 24));
          sub_969240(v14);
          sub_969240(v13);
          return a1;
        }
      }
    }
    sub_AAF450(a1, a2 + 32);
    return a1;
  }
  else
  {
    v6 = *(_DWORD *)(a2 + 40);
    *(_DWORD *)(a1 + 8) = v6;
    if ( v6 > 0x40 )
      sub_C43780(a1, (const void **)(a2 + 32));
    else
      *(_QWORD *)a1 = *(_QWORD *)(a2 + 32);
    v7 = *(_DWORD *)(a2 + 56);
    *(_DWORD *)(a1 + 24) = v7;
    if ( v7 <= 0x40 )
    {
      *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 48);
      return a1;
    }
    sub_C43780(a1 + 16, (const void **)(a2 + 48));
    return a1;
  }
}
