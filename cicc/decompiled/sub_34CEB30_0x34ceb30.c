// Function: sub_34CEB30
// Address: 0x34ceb30
//
__int64 __fastcall sub_34CEB30(__int64 a1, __int64 a2, char a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  unsigned int v6; // eax
  __int64 v7; // rdx
  _QWORD v9[3]; // [rsp+0h] [rbp-20h] BYREF

  v4 = sub_9208B0(*(_QWORD *)(a1 + 16), a2);
  v9[1] = v5;
  v9[0] = (unsigned __int64)(v4 + 7) >> 3;
  v6 = sub_CA1930(v9);
  v7 = 1LL << a3;
  LOBYTE(v7) = v6 != 0 && v6 <= (unsigned __int64)(1LL << a3);
  if ( (_BYTE)v7 )
  {
    LODWORD(v7) = v6 - 1;
    LOBYTE(v7) = (v6 & (v6 - 1)) == 0;
  }
  return (unsigned int)v7;
}
