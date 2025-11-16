// Function: sub_2E78040
// Address: 0x2e78040
//
unsigned __int64 __fastcall sub_2E78040(unsigned __int64 a1)
{
  unsigned __int64 v1; // r12
  unsigned __int64 i; // rax
  __int64 v3; // rdx
  __int64 v4; // rbx

  v1 = a1;
  for ( i = a1; (*(_BYTE *)(i + 44) & 8) != 0; i = *(_QWORD *)(i + 8) )
    ;
  v3 = *(_DWORD *)(a1 + 44) & 4;
  v4 = *(_QWORD *)(i + 8);
  if ( (_DWORD)v3 )
  {
    do
      v1 = *(_QWORD *)v1 & 0xFFFFFFFFFFFFFFF8LL;
    while ( (*(_BYTE *)(v1 + 44) & 4) != 0 );
  }
  while ( 1 )
  {
    if ( v4 == v1 )
      BUG();
    if ( (unsigned __int8)sub_2E88ED0(v1, 0, v3) )
      break;
    v1 = *(_QWORD *)(v1 + 8);
  }
  return v1;
}
