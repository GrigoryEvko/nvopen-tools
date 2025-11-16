// Function: sub_21DF3A0
// Address: 0x21df3a0
//
__int64 __fastcall sub_21DF3A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v10; // rax
  int v11; // edx
  __int64 v12; // rdx
  int v13; // ecx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int16 v16; // r12
  __int16 v17; // r12
  __int64 v18; // rsi
  _QWORD *v19; // r15
  __int64 v21; // [rsp+0h] [rbp-40h] BYREF
  int v22; // [rsp+8h] [rbp-38h]

  v6 = 0;
  v7 = *(_QWORD *)(a1 + 32);
  if ( *(_DWORD *)(v7 + 252) > 0x3Cu && *(_DWORD *)(v7 + 248) > 0x31u )
  {
    if ( *(_DWORD *)(a1 - 144) )
    {
      v8 = *(_QWORD *)(a2 + 32);
      v10 = *(_QWORD *)(v8 + 80);
      v11 = *(unsigned __int16 *)(v10 + 24);
      if ( v11 == 10 || v11 == 32 )
      {
        v12 = *(_QWORD *)(v8 + 160);
        v13 = *(unsigned __int16 *)(v12 + 24);
        if ( v13 == 10 || v13 == 32 )
        {
          v14 = *(_QWORD *)(v10 + 88);
          v6 = 0;
          if ( *(_DWORD *)(v14 + 32) == 1 )
          {
            v15 = *(_QWORD *)(v12 + 88);
            if ( *(_DWORD *)(v15 + 32) == 1 )
            {
              v16 = *(_QWORD *)(v15 + 24) == 1;
              if ( *(_QWORD *)(v14 + 24) == 1 )
                v17 = v16 + 391;
              else
                v17 = v16 + 389;
              v18 = *(_QWORD *)(a2 + 72);
              v19 = *(_QWORD **)(a1 - 176);
              v21 = v18;
              if ( v18 )
                sub_1623A60((__int64)&v21, v18, 2);
              v22 = *(_DWORD *)(a2 + 64);
              v6 = sub_1D2CD40(
                     v19,
                     v17,
                     (__int64)&v21,
                     5,
                     0,
                     a6,
                     *(_OWORD *)(v8 + 40),
                     *(_OWORD *)(v8 + 120),
                     *(_OWORD *)(v8 + 200));
              if ( v21 )
                sub_161E7C0((__int64)&v21, v21);
            }
          }
        }
        else
        {
          return 0;
        }
      }
    }
  }
  return v6;
}
