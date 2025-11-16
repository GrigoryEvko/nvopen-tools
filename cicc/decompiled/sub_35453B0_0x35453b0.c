// Function: sub_35453B0
// Address: 0x35453b0
//
__int64 __fastcall sub_35453B0(__int64 a1, _WORD *a2, int a3)
{
  __int64 v4; // rdx
  __int64 v7; // rcx
  unsigned __int16 *v8; // r11
  __int64 result; // rax
  unsigned __int16 *i; // rdi
  int v11; // esi
  int v12; // ecx
  int v13; // edx
  int v14; // ecx
  __int64 v15; // rax
  int v16; // esi
  int v17; // ecx
  int v18; // edx
  int v19; // ecx

  v4 = (unsigned __int16)a2[1];
  v7 = *(_QWORD *)(*(_QWORD *)a1 + 176LL);
  v8 = (unsigned __int16 *)(v7 + 6 * (v4 + (unsigned __int16)a2[2]));
  result = 3 * v4;
  for ( i = (unsigned __int16 *)(v7 + 6 * v4); v8 != i; i += 3 )
  {
    if ( i[1] )
    {
      v11 = a3;
      do
      {
        v12 = *(_DWORD *)(a1 + 480);
        v13 = v11 % v12;
        v14 = v11 % v12 + v12;
        if ( v13 >= 0 )
          v14 = v13;
        ++v11;
        v15 = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 144LL * v14);
        --*(_QWORD *)(v15 + 8LL * *i);
        result = a3 + (unsigned int)i[1];
      }
      while ( (int)result > v11 );
    }
  }
  if ( (*a2 & 0x1FFF) != 0 )
  {
    v16 = a3;
    do
    {
      v17 = *(_DWORD *)(a1 + 480);
      v18 = v16 % v17;
      v19 = v16 % v17 + v17;
      if ( v18 >= 0 )
        v19 = v18;
      ++v16;
      --*(_DWORD *)(*(_QWORD *)(a1 + 272) + 4LL * v19);
      result = a3 + (*a2 & 0x1FFFu);
    }
    while ( (int)result > v16 );
  }
  return result;
}
