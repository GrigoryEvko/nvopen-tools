// Function: sub_122F930
// Address: 0x122f930
//
__int64 __fastcall sub_122F930(__int64 a1, __int64 a2, __int64 *a3)
{
  int v4; // eax
  __int64 v5; // r9
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v9; // [rsp+0h] [rbp-80h]
  unsigned __int8 v10; // [rsp+Fh] [rbp-71h]
  __int64 *v11; // [rsp+18h] [rbp-68h] BYREF
  __int64 v12[4]; // [rsp+20h] [rbp-60h] BYREF
  char v13; // [rsp+40h] [rbp-40h]
  char v14; // [rsp+41h] [rbp-3Fh]

  v10 = sub_120AFE0(a1, 6, "expected '[' in catchpad/cleanuppad");
  if ( !v10 )
  {
    if ( *(_DWORD *)(a1 + 240) == 7 )
    {
LABEL_15:
      *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
    }
    else
    {
      v4 = *(_DWORD *)(a2 + 8);
      while ( !v4 || !(unsigned __int8)sub_120AFE0(a1, 4, "expected ',' in argument list") )
      {
        v11 = 0;
        v14 = 1;
        v12[0] = (__int64)"expected type";
        v13 = 3;
        if ( (unsigned __int8)sub_12190A0(a1, &v11, (int *)v12, 0) )
          break;
        if ( *((_BYTE *)v11 + 8) == 9 )
        {
          if ( (unsigned __int8)sub_12255B0((__int64 **)a1, v12, a3) )
            return 1;
        }
        else if ( (unsigned __int8)sub_1224B80((__int64 **)a1, (__int64)v11, v12, a3) )
        {
          return 1;
        }
        v6 = *(unsigned int *)(a2 + 8);
        v7 = v12[0];
        if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
        {
          v9 = v12[0];
          sub_C8D5F0(a2, (const void *)(a2 + 16), v6 + 1, 8u, v12[0], v5);
          v6 = *(unsigned int *)(a2 + 8);
          v7 = v9;
        }
        *(_QWORD *)(*(_QWORD *)a2 + 8 * v6) = v7;
        v4 = *(_DWORD *)(a2 + 8) + 1;
        *(_DWORD *)(a2 + 8) = v4;
        if ( *(_DWORD *)(a1 + 240) == 7 )
          goto LABEL_15;
      }
      return 1;
    }
  }
  return v10;
}
