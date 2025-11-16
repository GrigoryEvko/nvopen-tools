// Function: sub_155F620
// Address: 0x155f620
//
__int64 __fastcall sub_155F620(__int64 a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 v3; // r12
  __int64 v5[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = (__int64 *)(a2 + 24);
  v3 = a2 + 8LL * *(unsigned int *)(a2 + 16) + 24;
  if ( v3 == a2 + 24 )
  {
LABEL_6:
    *(_QWORD *)a1 = 0;
    *(_BYTE *)(a1 + 8) = 1;
    return a1;
  }
  else
  {
    while ( 1 )
    {
      v5[0] = *v2;
      if ( sub_155D460(v5, 2) )
        break;
      if ( (__int64 *)v3 == ++v2 )
        goto LABEL_6;
    }
    sub_155D750(a1, v5);
    return a1;
  }
}
