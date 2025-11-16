// Function: sub_2779A20
// Address: 0x2779a20
//
void __fastcall sub_2779A20(int *a1, __int64 a2, __int64 a3)
{
  bool v3; // zf
  __int64 v4; // rax
  int v5; // eax
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 *v8; // r12
  __int64 v9; // rax

  *((_QWORD *)a1 + 1) = 0;
  *((_QWORD *)a1 + 2) = 0;
  *((_BYTE *)a1 + 24) = 0;
  v3 = *(_BYTE *)a2 == 85;
  *a1 = 0;
  *((_QWORD *)a1 + 4) = a2;
  if ( v3 )
  {
    v4 = *(_QWORD *)(a2 - 32);
    if ( v4 )
    {
      if ( !*(_BYTE *)v4 && *(_QWORD *)(v4 + 24) == *(_QWORD *)(a2 + 80) && (*(_BYTE *)(v4 + 33) & 0x20) != 0 )
      {
        *a1 = *(_DWORD *)(v4 + 36);
        if ( !(unsigned __int8)sub_DFDD50(a3) )
        {
          v5 = *a1;
          if ( (*a1 & 0xFFFFFFFD) == 0xE4 )
          {
            if ( v5 == 228 )
            {
              if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
                v8 = *(__int64 **)(a2 - 8);
              else
                v8 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
              v9 = *v8;
              a1[5] = 65764;
              *((_BYTE *)a1 + 24) = 0;
              *((_QWORD *)a1 + 1) = v9;
            }
            else if ( v5 == 230 )
            {
              if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
                v6 = *(_QWORD *)(a2 - 8);
              else
                v6 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
              v7 = *(_QWORD *)(v6 + 32);
              a1[5] = (int)&loc_10000E4;
              *((_BYTE *)a1 + 24) = 0;
              *((_QWORD *)a1 + 1) = v7;
            }
          }
        }
      }
    }
  }
}
