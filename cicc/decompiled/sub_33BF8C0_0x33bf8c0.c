// Function: sub_33BF8C0
// Address: 0x33bf8c0
//
__int64 __fastcall sub_33BF8C0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  __int64 v3; // r13
  _QWORD *v4; // rbx
  int v5; // eax
  __int64 v6; // rsi
  unsigned __int8 *v7; // rsi
  __int64 v9; // rbx
  __int64 v10; // rdx
  __int64 v11; // r13
  _QWORD *v12; // rax
  __int64 v13; // rsi
  __int64 v14[2]; // [rsp+18h] [rbp-48h] BYREF
  _QWORD v15[7]; // [rsp+28h] [rbp-38h] BYREF

  v14[0] = a2;
  v2 = sub_337DC20(a1 + 8, v14);
  v3 = *v2;
  if ( *v2 )
  {
    v4 = v2;
    v5 = *(_DWORD *)(v3 + 24);
    if ( (unsigned int)(v5 - 11) <= 1 || (unsigned int)(v5 - 35) <= 1 )
    {
      v15[0] = 0;
      if ( (_QWORD *)(v3 + 80) != v15 )
      {
        v6 = *(_QWORD *)(v3 + 80);
        if ( v6 )
        {
          sub_B91220(v3 + 80, v6);
          v7 = (unsigned __int8 *)v15[0];
          *(_QWORD *)(v3 + 80) = v15[0];
          if ( v7 )
            sub_B976B0((__int64)v15, v7, v3 + 80);
        }
      }
    }
    return *v4;
  }
  else
  {
    v9 = sub_3389ED0(a1, v14[0]);
    v11 = v10;
    v12 = sub_337DC20(a1 + 8, v14);
    *v12 = v9;
    v13 = v14[0];
    *((_DWORD *)v12 + 2) = v11;
    sub_3380540(a1, v13, v9, v11);
    return v9;
  }
}
