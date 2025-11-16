// Function: sub_1593B40
// Address: 0x1593b40
//
void __fastcall sub_1593B40(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rdx
  unsigned __int64 v3; // rax
  __int64 v4; // rax

  if ( *a1 )
  {
    v2 = a1[1];
    v3 = a1[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v3 = v2;
    if ( v2 )
      *(_QWORD *)(v2 + 16) = *(_QWORD *)(v2 + 16) & 3LL | v3;
  }
  *a1 = a2;
  if ( a2 )
  {
    v4 = *(_QWORD *)(a2 + 8);
    a1[1] = v4;
    if ( v4 )
      *(_QWORD *)(v4 + 16) = (unsigned __int64)(a1 + 1) | *(_QWORD *)(v4 + 16) & 3LL;
    a1[2] = (a2 + 8) | a1[2] & 3LL;
    *(_QWORD *)(a2 + 8) = a1;
  }
}
