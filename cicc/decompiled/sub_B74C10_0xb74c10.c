// Function: sub_B74C10
// Address: 0xb74c10
//
void __fastcall sub_B74C10(__int64 a1)
{
  __int64 v1; // r12
  _QWORD *v2; // rbx
  _QWORD *v3; // r12
  __int64 v4; // r13

  v1 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v1 )
  {
    v2 = *(_QWORD **)(a1 + 8);
    v3 = &v2[2 * v1];
    do
    {
      if ( *v2 != -4096 && *v2 != -8192 )
      {
        v4 = v2[1];
        if ( v4 )
        {
          sub_BD7260(v2[1]);
          sub_BD2DD0(v4);
        }
      }
      v2 += 2;
    }
    while ( v3 != v2 );
  }
}
