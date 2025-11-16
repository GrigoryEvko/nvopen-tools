// Function: sub_667320
// Address: 0x667320
//
__int64 __fastcall sub_667320(__int64 *a1)
{
  __int64 *v1; // r12
  _QWORD *v3; // rax
  __int64 v4; // rax
  __int64 result; // rax
  __int64 *v6; // rsi
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rsi

  v1 = (__int64 *)a1[56];
  if ( *((_BYTE *)v1 + 16) == 2 )
  {
    v3 = a1 + 56;
  }
  else
  {
    do
    {
      v3 = v1;
      v1 = (__int64 *)*v1;
    }
    while ( *((_BYTE *)v1 + 16) != 2 );
  }
  *v3 = *v1;
  v4 = *a1;
  if ( !*a1 || *(_BYTE *)(v4 + 80) != 11 )
    return sub_6851C0(2529, v1 + 1);
  if ( (unsigned int)sub_72F8B0(*(_QWORD *)(v4 + 88)) )
  {
    v6 = v1 + 1;
    if ( !HIDWORD(qword_4F077B4) )
      return sub_6851C0(2529, v6);
    sub_684B30(2529, v6);
  }
  v7 = *(_QWORD *)(*a1 + 88);
  v8 = *(_QWORD *)(v7 + 152);
  if ( *(_BYTE *)(v8 + 140) == 7 )
    *(_BYTE *)(*(_QWORD *)(v8 + 168) + 20LL) |= 1u;
  if ( (*((_BYTE *)a1 + 122) & 1) != 0 )
  {
    v9 = *(_QWORD *)(v7 + 72);
    return sub_7294B0(v1, v9 + 48);
  }
  result = dword_4F04C3C;
  if ( !dword_4F04C3C )
  {
    result = sub_86A2A0(v7);
    if ( result )
    {
      v9 = *(_QWORD *)(*(_QWORD *)(result + 24) + 8LL);
      return sub_7294B0(v1, v9 + 48);
    }
  }
  return result;
}
