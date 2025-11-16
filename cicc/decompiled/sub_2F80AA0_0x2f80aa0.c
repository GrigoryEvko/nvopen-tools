// Function: sub_2F80AA0
// Address: 0x2f80aa0
//
__int64 __fastcall sub_2F80AA0(__int64 *a1, __int64 a2)
{
  __int64 (*v2)(); // rdx
  __int64 v3; // rax
  __int64 v4; // rax
  unsigned int v5; // r14d
  __int64 v6; // r13
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rsi

  v2 = *(__int64 (**)())(*(_QWORD *)a2 + 128LL);
  v3 = 0;
  if ( v2 != sub_2DAC790 )
    v3 = ((__int64 (__fastcall *)(__int64))v2)(a2);
  a1[2] = v3;
  v4 = a1[1];
  v5 = 0;
  v6 = *(unsigned int *)(v4 + 64);
  if ( *(_DWORD *)(v4 + 64) )
  {
    v7 = 0;
    do
    {
      v8 = v7 & 0x7FFFFFFF;
      if ( *(_DWORD *)(*a1 + 160) > (unsigned int)v8 )
      {
        v9 = *(_QWORD *)(*(_QWORD *)(*a1 + 152) + 8 * v8);
        if ( v9 )
        {
          if ( *(_QWORD *)(v9 + 104) )
            v5 |= sub_2F7EA70(a1, v9);
        }
      }
      ++v7;
    }
    while ( v6 != v7 );
  }
  return v5;
}
