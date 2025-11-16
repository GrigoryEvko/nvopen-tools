// Function: sub_34F5E30
// Address: 0x34f5e30
//
__int64 __fastcall sub_34F5E30(_QWORD *a1, __int64 *a2, __int64 a3)
{
  __int64 *v3; // rax
  __int64 v4; // r8
  unsigned __int64 v5; // rdx
  __int64 v6; // rcx
  unsigned __int64 v7; // rcx
  __int64 result; // rax

  *a1 = a3;
  v3 = a2;
  a1[1] = a2;
  if ( a2 == *(__int64 **)(a3 + 56) )
  {
    a1[2] = a3 + 48;
    if ( a2 )
    {
      v4 = *a2;
      goto LABEL_9;
    }
LABEL_15:
    BUG();
  }
  v4 = *a2;
  v5 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v5 )
    goto LABEL_15;
  v6 = *(_QWORD *)v5;
  if ( (*(_QWORD *)v5 & 4) == 0 && (*(_BYTE *)(v5 + 44) & 4) != 0 )
  {
    while ( 1 )
    {
      v7 = v6 & 0xFFFFFFFFFFFFFFF8LL;
      v5 = v7;
      if ( (*(_BYTE *)(v7 + 44) & 4) == 0 )
        break;
      v6 = *(_QWORD *)v7;
    }
  }
  a1[2] = v5;
LABEL_9:
  if ( (v4 & 4) == 0 && (*((_BYTE *)a2 + 44) & 8) != 0 )
  {
    do
      v3 = (__int64 *)v3[1];
    while ( (*((_BYTE *)v3 + 44) & 8) != 0 );
  }
  result = v3[1];
  a1[3] = result;
  return result;
}
