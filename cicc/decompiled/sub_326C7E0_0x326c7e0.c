// Function: sub_326C7E0
// Address: 0x326c7e0
//
__int64 __fastcall sub_326C7E0(__int64 *a1, unsigned __int16 a2, __int64 a3, __int64 a4)
{
  __int64 (*v4)(); // rax
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rcx
  _QWORD *v10; // rdx
  int v11; // eax

  v4 = *(__int64 (**)())(*(_QWORD *)a4 + 1240LL);
  if ( v4 == sub_2FE3380 )
    return 0;
  result = ((__int64 (__fastcall *)(__int64))v4)(a4);
  if ( !(_BYTE)result )
    return 0;
  v7 = *a1;
  if ( *(_DWORD *)(*a1 + 24) == 208 )
  {
    v8 = *(_QWORD *)(v7 + 56);
    if ( v8 )
    {
      if ( !*(_QWORD *)(v8 + 32) )
      {
        v9 = 1;
        if ( a2 == 1 || a2 && (v9 = a2, *(_QWORD *)(a4 + 8LL * a2 + 112)) )
        {
          if ( (*(_BYTE *)(a4 + 500 * v9 + 6621) & 0xFB) == 0 )
          {
            v10 = *(_QWORD **)(v7 + 40);
            v11 = *(_DWORD *)(v10[10] + 96LL);
            if ( v11 != 20 )
            {
              if ( v11 == 18 )
                return sub_33E07E0(v10[5], v10[6], 0);
              return 0;
            }
            return sub_33E0720(v10[5], v10[6], 0);
          }
        }
      }
    }
  }
  return result;
}
