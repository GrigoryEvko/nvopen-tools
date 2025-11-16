// Function: sub_1E74370
// Address: 0x1e74370
//
void __fastcall sub_1E74370(__int64 a1, __int64 a2, __int64 a3, char a4, int a5, int a6)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v10; // r8
  int v11; // r10d
  __int64 v12; // rsi
  __int64 v13; // r9
  unsigned __int64 v14; // rdx

  *(_QWORD *)(a2 + 16) = a3;
  *(_BYTE *)(a2 + 25) = a4;
  v6 = *(_QWORD *)(a1 + 128);
  if ( *(_BYTE *)(v6 + 2568) )
  {
    v7 = *(_QWORD *)(v6 + 2576);
    v10 = *(_QWORD *)(v6 + 3064);
    v11 = a2 + 26;
    v12 = *(_QWORD *)(a3 + 8);
    v13 = (*(_QWORD *)(v6 + 3072) - v10) >> 2;
    if ( a4 )
    {
      sub_1EE9490(a6, v12, v11, v10, v13, v13, v7);
    }
    else
    {
      v14 = *(_QWORD *)(v6 + 2552) + ((unsigned __int64)*(unsigned int *)(a3 + 192) << 6);
      if ( byte_4FC7920 )
        sub_1EE7CA0(a6, v12, v14, v11, v10, v13, v7);
      else
        sub_1EE7EC0(a5, v12, v14, v11, v10, v13, v7);
    }
  }
}
