// Function: sub_2E46590
// Address: 0x2e46590
//
bool __fastcall sub_2E46590(int *a1, int *a2)
{
  __int16 *v3; // rdi
  __int16 *v4; // rax
  int v5; // ecx
  int v6; // edx
  int v7; // r8d
  __int16 v8; // si

  v3 = (__int16 *)*((_QWORD *)a1 + 4);
  v4 = (__int16 *)*((_QWORD *)a1 + 1);
  v5 = *a1;
  v6 = *((unsigned __int16 *)a1 + 8);
  if ( v3 != v4 )
  {
    v7 = *a2;
    do
    {
      if ( v6 == v7 )
        break;
      v8 = *v4++;
      LOWORD(v5) = v8 + v5;
      if ( !v8 )
        v4 = 0;
      v6 = (unsigned __int16)v5;
    }
    while ( v3 != v4 );
  }
  return v3 != v4;
}
