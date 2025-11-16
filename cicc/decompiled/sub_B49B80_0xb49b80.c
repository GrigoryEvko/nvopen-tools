// Function: sub_B49B80
// Address: 0xb49b80
//
__int64 __fastcall sub_B49B80(__int64 a1, int a2, int a3)
{
  unsigned int v4; // r12d
  __int64 v5; // rax
  unsigned int v7; // eax
  int v8; // eax
  int v9; // eax
  _QWORD v10[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = sub_A74710((_QWORD *)(a1 + 72), a2 + 1, a3);
  if ( !(_BYTE)v4 )
  {
    v5 = *(_QWORD *)(a1 - 32);
    if ( v5 )
    {
      if ( !*(_BYTE *)v5 && *(_QWORD *)(v5 + 24) == *(_QWORD *)(a1 + 80) )
      {
        v10[0] = *(_QWORD *)(v5 + 120);
        v7 = sub_A74710(v10, a2 + 1, a3);
        if ( (_BYTE)v7 )
        {
          if ( a3 != 51 )
          {
            if ( a3 == 78 )
            {
              LOBYTE(v9) = sub_B49990(a1);
              return v9 ^ 1u;
            }
            if ( a3 != 50 )
              return v7;
            if ( sub_B49990(a1) )
              return v4;
          }
          LOBYTE(v8) = sub_B49A80(a1);
          return v8 ^ 1u;
        }
      }
    }
  }
  return v4;
}
