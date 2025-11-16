// Function: sub_1AEC550
// Address: 0x1aec550
//
__int64 __fastcall sub_1AEC550(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // r15
  _QWORD *v7; // r14
  _QWORD *v8; // rax
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  unsigned int v13; // [rsp+Ch] [rbp-34h]

  v4 = *(_QWORD **)(a1 + 8);
  if ( v4 )
  {
    v13 = 0;
    do
    {
      v7 = v4;
      v4 = (_QWORD *)v4[1];
      v8 = sub_1648700((__int64)v7);
      if ( sub_15CC890(a3, a4, v8[5]) )
      {
        if ( *v7 )
        {
          v9 = v7[1];
          v10 = v7[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v10 = v9;
          if ( v9 )
            *(_QWORD *)(v9 + 16) = *(_QWORD *)(v9 + 16) & 3LL | v10;
        }
        *v7 = a2;
        if ( a2 )
        {
          v11 = *(_QWORD *)(a2 + 8);
          v7[1] = v11;
          if ( v11 )
            *(_QWORD *)(v11 + 16) = (unsigned __int64)(v7 + 1) | *(_QWORD *)(v11 + 16) & 3LL;
          v7[2] = (a2 + 8) | v7[2] & 3LL;
          *(_QWORD *)(a2 + 8) = v7;
        }
        ++v13;
      }
    }
    while ( v4 );
  }
  else
  {
    return 0;
  }
  return v13;
}
