// Function: sub_2BE2160
// Address: 0x2be2160
//
void __fastcall sub_2BE2160(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  int v4; // r12d
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rdx

  v3 = a1[9] + 16 * a3;
  v4 = *(_DWORD *)(v3 + 8);
  v5 = *(_QWORD *)v3;
  v6 = *(_QWORD *)(a1[7] + 56LL) + 48 * a3;
  if ( v4 && v5 == a1[3] )
  {
    if ( v4 <= 1 )
    {
      *(_DWORD *)(v3 + 8) = v4 + 1;
      sub_2BE1DD0((__int64)a1, a2, *(_QWORD *)(v6 + 16));
      --*(_DWORD *)(v3 + 8);
    }
  }
  else
  {
    v7 = a1[3];
    *(_DWORD *)(v3 + 8) = 1;
    *(_QWORD *)v3 = v7;
    sub_2BE1DD0((__int64)a1, a2, *(_QWORD *)(v6 + 16));
    *(_QWORD *)v3 = v5;
    *(_DWORD *)(v3 + 8) = v4;
  }
}
