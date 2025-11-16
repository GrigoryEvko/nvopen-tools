// Function: sub_1607030
// Address: 0x1607030
//
void __fastcall sub_1607030(__int64 a1)
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
      if ( *v2 != -8 && *v2 != -16 )
      {
        v4 = v2[1];
        if ( v4 )
        {
          sub_164BE60(v2[1]);
          sub_1648B90(v4);
        }
      }
      v2 += 2;
    }
    while ( v3 != v2 );
  }
}
