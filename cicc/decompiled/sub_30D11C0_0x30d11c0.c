// Function: sub_30D11C0
// Address: 0x30d11c0
//
void __fastcall sub_30D11C0(_DWORD *a1, int a2, unsigned int a3, unsigned __int8 a4)
{
  int v4; // r8d
  int v5; // eax

  v4 = qword_5030168;
  if ( a2 )
  {
    v5 = 2 * qword_5030168;
    if ( !a4 )
      a1[172] += v5;
    a1[170] += v5 + v4 * a2;
  }
  else if ( a3 <= 3 )
  {
    a1[171] += 2 * qword_5030168 * (a3 - a4);
  }
  else
  {
    a1[173] += 2 * qword_5030168 * (3LL * (int)a3 / 2 - 1);
  }
}
