// Function: sub_CB77B0
// Address: 0xcb77b0
//
__int64 __fastcall sub_CB77B0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v4; // rdx
  signed __int64 v5; // rax
  __int64 v7; // r13
  signed __int64 v8; // rsi
  __int64 v9; // rdx

  v3 = a1[5];
  v4 = a3 - a2;
  if ( v4 )
  {
    v5 = a1[4];
    v7 = v4;
    v8 = v5 + v4;
    v9 = a1[5];
    if ( v5 < v8 )
    {
      sub_CB7740((__int64)a1, v8);
      v9 = a1[5];
    }
    memmove((void *)(a1[3] + 8 * v9), (const void *)(a1[3] + 8 * a2), 8 * v7);
    a1[5] += v7;
  }
  return v3;
}
