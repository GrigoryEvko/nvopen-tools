// Function: sub_2D04510
// Address: 0x2d04510
//
__int64 __fastcall sub_2D04510(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r11
  __int64 v5; // r10
  __int64 i; // r9
  __int64 v8; // r8
  __int64 v9; // rax
  __int64 v10; // rdx

  v4 = a2;
  v5 = (a3 - 1) / 2;
  if ( a2 < v5 )
  {
    for ( i = a2; ; i = a2 )
    {
      a2 = 2 * i + 2;
      v8 = 2 * i + 1;
      v9 = *(_QWORD *)(a1 + 8 * a2);
      v10 = *(_QWORD *)(a1 + 8 * v8);
      if ( (float)((float)(100 * *(_DWORD *)(v10 + 36)
                         + *(_DWORD *)(v10 + 16)
                         + 100 * *(unsigned __int8 *)(v10 + 40)
                         + 10 * *(_DWORD *)(v10 + 32))
                 / (float)(*(_DWORD *)(v10 + 24) + *(_DWORD *)(v10 + 20) + *(_DWORD *)(v10 + 28) + 1)) > (float)((float)(100 * *(_DWORD *)(v9 + 36) + *(_DWORD *)(v9 + 16) + 100 * *(unsigned __int8 *)(v9 + 40) + 10 * *(_DWORD *)(v9 + 32)) / (float)(*(_DWORD *)(v9 + 24) + *(_DWORD *)(v9 + 20) + *(_DWORD *)(v9 + 28) + 1)) )
      {
        v9 = *(_QWORD *)(a1 + 8 * v8);
        a2 = 2 * i + 1;
      }
      *(_QWORD *)(a1 + 8 * i) = v9;
      if ( a2 >= v5 )
        break;
    }
  }
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == a2 )
  {
    *(_QWORD *)(a1 + 8 * a2) = *(_QWORD *)(a1 + 8 * (2 * a2 + 1));
    a2 = 2 * a2 + 1;
  }
  return sub_2D04420(a1, a2, v4, a4);
}
