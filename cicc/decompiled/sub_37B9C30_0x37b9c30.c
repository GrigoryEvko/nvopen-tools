// Function: sub_37B9C30
// Address: 0x37b9c30
//
__int64 __fastcall sub_37B9C30(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx

  v2 = *a2;
  v3 = *a2 + 48LL * *((unsigned int *)a2 + 2);
  if ( *a2 != v3 )
  {
    do
    {
      if ( !*(_BYTE *)(v2 + 40) )
        break;
      v2 += 48;
    }
    while ( v3 != v2 );
  }
  *(_QWORD *)a1 = v2;
  *(_QWORD *)(a1 + 8) = v3;
  *(_BYTE *)(a1 + 25) = 1;
  *(_QWORD *)(a1 + 32) = v3;
  *(_QWORD *)(a1 + 40) = v3;
  *(_BYTE *)(a1 + 57) = 1;
  return a1;
}
