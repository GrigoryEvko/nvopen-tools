// Function: sub_2D5B750
// Address: 0x2d5b750
//
__int64 __fastcall sub_2D5B750(unsigned __int16 *a1)
{
  unsigned __int16 v1; // ax
  __int64 v3; // rax
  char v4; // cl
  __int64 v5; // rax
  __int64 v6; // [rsp-8h] [rbp-8h]

  v1 = *a1;
  if ( !*a1 )
    return sub_3007260(a1);
  if ( v1 == 1 || (unsigned __int16)(v1 - 504) <= 7u )
    BUG();
  v3 = 16LL * (v1 - 1);
  v4 = byte_444C4A0[v3 + 8];
  v5 = *(_QWORD *)&byte_444C4A0[v3];
  *((_BYTE *)&v6 - 8) = v4;
  *(&v6 - 2) = v5;
  return *(&v6 - 2);
}
