// Function: sub_B91A30
// Address: 0xb91a30
//
void __fastcall sub_B91A30(__int64 a1)
{
  __int64 *v1; // r13
  __int64 v2; // r12
  __int64 v3; // rbx
  __int64 v4; // rsi

  v1 = *(__int64 **)(a1 + 56);
  v2 = *v1;
  v3 = *v1 + 8LL * *((unsigned int *)v1 + 2);
  while ( v2 != v3 )
  {
    while ( 1 )
    {
      v4 = *(_QWORD *)(v3 - 8);
      v3 -= 8;
      if ( !v4 )
        break;
      sub_B91220(v3, v4);
      if ( v2 == v3 )
        goto LABEL_5;
    }
  }
LABEL_5:
  *((_DWORD *)v1 + 2) = 0;
}
