// Function: sub_2A693D0
// Address: 0x2a693d0
//
void __fastcall sub_2A693D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  _BYTE *v3; // r12
  __int64 v4; // rax
  __int64 v5; // rcx
  unsigned int v6; // edx
  __int64 v7; // r13
  _BYTE *v8; // rsi
  unsigned __int8 *v9; // rax
  int v10; // r9d
  unsigned __int8 v11[104]; // [rsp-68h] [rbp-68h] BYREF

  v2 = *(_QWORD *)(a2 - 64);
  if ( *(_BYTE *)(*(_QWORD *)(v2 + 8) + 8LL) != 15 )
  {
    if ( *(_DWORD *)(a1 + 216) )
    {
      v3 = *(_BYTE **)(a2 - 32);
      if ( *v3 == 3 )
      {
        v4 = *(unsigned int *)(a1 + 224);
        v5 = *(_QWORD *)(a1 + 208);
        if ( (_DWORD)v4 )
        {
          v6 = (v4 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
          v7 = v5 + 48LL * v6;
          v8 = *(_BYTE **)v7;
          if ( v3 == *(_BYTE **)v7 )
          {
LABEL_8:
            if ( v7 != v5 + 48 * v4 )
            {
              v9 = (unsigned __int8 *)sub_2A68BC0(a1, (unsigned __int8 *)v2);
              sub_22C05A0((__int64)v11, v9);
              sub_2A639B0(a1, (_BYTE *)(v7 + 8), (__int64)v3, (__int64)v11, 0x100000000LL);
              sub_22C0090(v11);
              if ( *(_BYTE *)(v7 + 8) == 6 )
              {
                sub_22C0090((unsigned __int8 *)(v7 + 8));
                *(_QWORD *)v7 = -8192;
                --*(_DWORD *)(a1 + 216);
                ++*(_DWORD *)(a1 + 220);
              }
            }
          }
          else
          {
            v10 = 1;
            while ( v8 != (_BYTE *)-4096LL )
            {
              v6 = (v4 - 1) & (v10 + v6);
              v7 = v5 + 48LL * v6;
              v8 = *(_BYTE **)v7;
              if ( v3 == *(_BYTE **)v7 )
                goto LABEL_8;
              ++v10;
            }
          }
        }
      }
    }
  }
}
