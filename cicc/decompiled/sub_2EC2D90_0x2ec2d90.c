// Function: sub_2EC2D90
// Address: 0x2ec2d90
//
__int64 __fastcall sub_2EC2D90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v6; // rax
  __int64 v7; // rdi
  char v8; // al
  __int64 v9; // rdi
  __int64 result; // rax

  sub_2EC1090(a1, a2, a3, a4, a5);
  v6 = *(_QWORD *)(a1 + 920);
  if ( v6 != a2 + 48 )
  {
    if ( !v6 )
      BUG();
    if ( (*(_BYTE *)v6 & 4) == 0 && (*(_BYTE *)(v6 + 44) & 8) != 0 )
    {
      do
        v6 = *(_QWORD *)(v6 + 8);
      while ( (*(_BYTE *)(v6 + 44) & 8) != 0 );
    }
    v6 = *(_QWORD *)(v6 + 8);
  }
  v7 = *(_QWORD *)(a1 + 3472);
  *(_QWORD *)(a1 + 3632) = v6;
  *(_DWORD *)(a1 + 4008) = 0;
  v8 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v7 + 48LL))(v7);
  v9 = *(_QWORD *)(a1 + 3472);
  *(_BYTE *)(a1 + 4016) = v8;
  result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v9 + 56LL))(v9);
  *(_BYTE *)(a1 + 4017) = result;
  return result;
}
