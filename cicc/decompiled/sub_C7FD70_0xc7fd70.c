// Function: sub_C7FD70
// Address: 0xc7fd70
//
__int64 __fastcall sub_C7FD70(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  __int64 v7; // r9
  __int64 v8; // rdx
  __int64 v9; // r10
  __int64 v10; // r11
  int v11; // eax
  __int64 v12; // r12
  __int64 v13; // rcx
  __int64 v14; // rdi
  __int64 v15; // r8
  __int64 v16; // rsi

  if ( (_DWORD)a1 )
  {
    sub_2241E50(a1, a2, a3, a4, a5);
    result = (unsigned int)*__errno_location();
    *(_QWORD *)a3 = 0;
    *(_QWORD *)(a3 + 8) = 0;
    *(_QWORD *)(a3 + 16) = 0;
    *(_QWORD *)(a3 + 24) = 0;
    *(_QWORD *)(a3 + 32) = 0;
    *(_QWORD *)(a3 + 48) = 0;
    if ( (_DWORD)result == 2 )
      *(_QWORD *)(a3 + 40) = 0xFFFF00000001LL;
    else
      *(_QWORD *)(a3 + 40) = 0xFFFF00000000LL;
    *(_QWORD *)(a3 + 56) = 0;
    *(_QWORD *)(a3 + 64) = 0;
  }
  else
  {
    v7 = a2[6];
    v8 = 3;
    v9 = *(__int64 *)((char *)a2 + 28);
    v10 = a2[11];
    v11 = a2[3] & 0xF000;
    v12 = a2[9];
    v13 = a2[1];
    v14 = a2[2];
    v15 = a2[3] & 0xFFF;
    v16 = *a2;
    if ( v11 != 0x4000 )
    {
      v8 = 2;
      if ( v11 != 0x8000 )
      {
        v8 = 5;
        if ( v11 != 24576 )
        {
          v8 = 6;
          if ( v11 != 0x2000 )
          {
            v8 = 7;
            if ( v11 != 4096 )
            {
              v8 = 8;
              if ( v11 != 49152 )
                v8 = 5 * (unsigned int)(v11 != 40960) + 4;
            }
          }
        }
      }
    }
    *(_QWORD *)a3 = v12;
    *(_DWORD *)(a3 + 40) = v8;
    *(_QWORD *)(a3 + 8) = v10;
    *(_QWORD *)(a3 + 16) = 0;
    *(_QWORD *)(a3 + 24) = v9;
    *(_QWORD *)(a3 + 32) = v7;
    *(_DWORD *)(a3 + 44) = v15;
    *(_QWORD *)(a3 + 48) = v16;
    *(_QWORD *)(a3 + 56) = v14;
    *(_QWORD *)(a3 + 64) = v13;
    sub_2241E40(v14, v16, v8, v13, v15);
    return 0;
  }
  return result;
}
