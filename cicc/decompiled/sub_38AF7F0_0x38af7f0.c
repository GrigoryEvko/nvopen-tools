// Function: sub_38AF7F0
// Address: 0x38af7f0
//
__int64 __fastcall sub_38AF7F0(__int64 a1, __int64 a2, __int64 *a3, double a4, double a5, double a6)
{
  int v7; // eax
  int v8; // r8d
  int v9; // r9d
  __int64 v10; // rax
  unsigned __int8 v12; // [rsp+Fh] [rbp-61h]
  __int64 v13; // [rsp+18h] [rbp-58h] BYREF
  __int64 v14[2]; // [rsp+20h] [rbp-50h] BYREF
  char v15; // [rsp+30h] [rbp-40h]
  char v16; // [rsp+31h] [rbp-3Fh]

  v12 = sub_388AF10(a1, 6, "expected '[' in catchpad/cleanuppad");
  if ( !v12 )
  {
    if ( *(_DWORD *)(a1 + 64) == 7 )
    {
LABEL_15:
      *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
    }
    else
    {
      v7 = *(_DWORD *)(a2 + 8);
      while ( !v7 || !(unsigned __int8)sub_388AF10(a1, 4, "expected ',' in argument list") )
      {
        v13 = 0;
        v16 = 1;
        v14[0] = (__int64)"expected type";
        v15 = 3;
        if ( (unsigned __int8)sub_3891B00(a1, &v13, (__int64)v14, 0) )
          break;
        if ( *(_BYTE *)(v13 + 8) == 8 )
        {
          if ( (unsigned __int8)sub_38A2200((__int64 **)a1, v14, a3, a4, a5, a6) )
            return 1;
        }
        else if ( (unsigned __int8)sub_38A1070((__int64 **)a1, v13, v14, a3, a4, a5, a6) )
        {
          return 1;
        }
        v10 = *(unsigned int *)(a2 + 8);
        if ( (unsigned int)v10 >= *(_DWORD *)(a2 + 12) )
        {
          sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, v8, v9);
          v10 = *(unsigned int *)(a2 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a2 + 8 * v10) = v14[0];
        v7 = *(_DWORD *)(a2 + 8) + 1;
        *(_DWORD *)(a2 + 8) = v7;
        if ( *(_DWORD *)(a1 + 64) == 7 )
          goto LABEL_15;
      }
      return 1;
    }
  }
  return v12;
}
