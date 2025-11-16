// Function: sub_25591C0
// Address: 0x25591c0
//
__int64 __fastcall sub_25591C0(__int64 *a1, unsigned __int8 *a2)
{
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rdx
  __int64 v7; // rcx
  unsigned __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rax
  char v12; // dl
  __int64 v13; // r13
  __int64 v14; // rdi
  char v15; // al
  __int64 v16; // rsi
  _BYTE v17[80]; // [rsp+10h] [rbp-50h] BYREF

  if ( (!sub_B46560(a2) || !(unsigned __int8)sub_B46490((__int64)a2))
    && !(unsigned __int8)sub_B19060(*a1 + 200, (__int64)a2, v4, v5)
    && !(unsigned __int8)sub_B19060(*a1 + 104, (__int64)a2, v6, v7) )
  {
    v9 = sub_2537AD0(a2);
    v11 = sub_2559000(*a1, a1[1], v9, v10);
    if ( v12 == 1 )
    {
      if ( v11 )
      {
        if ( *(_BYTE *)v11 == 20 )
        {
          v13 = *(_QWORD *)(v11 + 8);
          v14 = sub_B43CB0((__int64)a2);
          if ( (unsigned int)*(unsigned __int8 *)(v13 + 8) - 17 <= 1 )
            v13 = **(_QWORD **)(v13 + 16);
          v15 = sub_B2F070(v14, *(_DWORD *)(v13 + 8) >> 8);
          v16 = *a1;
          if ( v15 )
            sub_BED950((__int64)v17, v16 + 200, (__int64)a2);
          else
            sub_BED950((__int64)v17, v16 + 104, (__int64)a2);
        }
        else
        {
          sub_BED950((__int64)v17, *a1 + 200, (__int64)a2);
        }
      }
    }
  }
  return 1;
}
