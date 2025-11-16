// Function: sub_2FDBA40
// Address: 0x2fdba40
//
__int64 __fastcall sub_2FDBA40(__int64 a1)
{
  __int64 v1; // rax
  unsigned int v2; // r14d
  __int64 *v3; // r15
  char v4; // r12
  __int64 *v5; // r13
  __int64 *v7; // [rsp+8h] [rbp-38h]

  if ( *(_BYTE *)(a1 + 56) && (_BYTE)qword_5026588 )
    sub_2FDB0A0(*(_QWORD *)(a1 + 32), 1);
  v1 = *(_QWORD *)(a1 + 32);
  v2 = 0;
  v7 = (__int64 *)(v1 + 320);
  v3 = *(__int64 **)(*(_QWORD *)(v1 + 328) + 8LL);
  if ( (__int64 *)(v1 + 320) != v3 )
  {
    do
    {
      v5 = v3;
      v3 = (__int64 *)v3[1];
      if ( !(_DWORD)qword_50264A8 )
        break;
      v4 = sub_2FD62C0((__int64)v5);
      if ( (unsigned __int8)sub_2FD64C0((__int64 *)a1, v4, v5) )
        v2 |= sub_2FDA680(a1, v4, (__int64)v5, 0, 0, 0, 0);
    }
    while ( v3 != v7 );
  }
  if ( *(_BYTE *)(a1 + 56) && (_BYTE)qword_5026588 )
    sub_2FDB0A0(*(_QWORD *)(a1 + 32), 0);
  return v2;
}
