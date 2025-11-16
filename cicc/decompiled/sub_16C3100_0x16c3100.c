// Function: sub_16C3100
// Address: 0x16c3100
//
__int64 __fastcall sub_16C3100(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  int v7; // eax
  __int64 v8; // rdx
  int v9; // ecx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r10
  __int64 v13; // r11
  __int64 v14; // rcx
  __int64 v15; // rdi
  __int64 v16; // rsi

  if ( (_DWORD)a1 )
  {
    sub_2241E50(a1, a2, a3, a4, a5);
    result = (unsigned int)*__errno_location();
    *(_QWORD *)a3 = 0;
    *(_QWORD *)(a3 + 8) = 0;
    *(_QWORD *)(a3 + 16) = 0;
    *(_QWORD *)(a3 + 24) = 0;
    *(_QWORD *)(a3 + 40) = 0;
    if ( (_DWORD)result == 2 )
      *(_QWORD *)(a3 + 32) = 0xFFFF00000001LL;
    else
      *(_QWORD *)(a3 + 32) = 0xFFFF00000000LL;
    *(_QWORD *)(a3 + 48) = 0;
    *(_QWORD *)(a3 + 56) = 0;
  }
  else
  {
    v7 = *((_DWORD *)a2 + 6);
    v8 = 3;
    v9 = v7 & 0xF000;
    if ( v9 != 0x4000 )
    {
      v8 = 2;
      if ( v9 != 0x8000 )
      {
        v8 = 5;
        if ( v9 != 24576 )
        {
          v8 = 6;
          if ( v9 != 0x2000 )
          {
            v8 = 7;
            if ( v9 != 4096 )
            {
              v8 = 8;
              if ( v9 != 49152 )
                v8 = 5 * (unsigned int)(v9 != 40960) + 4;
            }
          }
        }
      }
    }
    v10 = a2[6];
    v11 = *(__int64 *)((char *)a2 + 28);
    v12 = a2[11];
    v13 = a2[9];
    v14 = a2[1];
    v15 = a2[2];
    v16 = *a2;
    *(_DWORD *)(a3 + 32) = v8;
    *(_QWORD *)a3 = v13;
    *(_QWORD *)(a3 + 8) = v12;
    *(_QWORD *)(a3 + 16) = v11;
    *(_QWORD *)(a3 + 24) = v10;
    *(_DWORD *)(a3 + 36) = v7 & 0xFFF;
    *(_QWORD *)(a3 + 40) = v16;
    *(_QWORD *)(a3 + 48) = v15;
    *(_QWORD *)(a3 + 56) = v14;
    sub_2241E40(v15, v16, v8, v14, v10);
    return 0;
  }
  return result;
}
