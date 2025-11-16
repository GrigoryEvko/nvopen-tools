// Function: sub_E47BC0
// Address: 0xe47bc0
//
char __fastcall sub_E47BC0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  int v4; // edx
  __int64 v5; // rsi
  int v6; // ecx
  unsigned int v7; // edx
  __int64 v8; // rdi
  int v9; // r8d
  __int64 v10; // r14
  __int64 v11; // r15
  bool v12; // zf
  __int64 v13; // rax
  unsigned __int8 *v14; // r13
  __int64 v16; // [rsp+8h] [rbp-68h]
  _BYTE v17[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v18; // [rsp+30h] [rbp-40h]

  v3 = sub_B326A0(a1);
  if ( v3 )
  {
    v4 = *(_DWORD *)(a2 + 24);
    v5 = *(_QWORD *)(a2 + 8);
    if ( v4 )
    {
      v6 = v4 - 1;
      v7 = (v4 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v8 = *(_QWORD *)(v5 + 8LL * v7);
      if ( v3 == v8 )
      {
LABEL_4:
        if ( *(_QWORD *)(a1 + 16) )
        {
          if ( *(_BYTE *)a1 )
          {
            if ( *(_BYTE *)a1 == 3 )
            {
              LOBYTE(v3) = sub_B30160(a1, 0);
            }
            else
            {
              v10 = *(_QWORD *)(a1 + 24);
              v11 = *(_QWORD *)(a1 + 40);
              v12 = *(_BYTE *)(v10 + 8) == 13;
              v18 = 257;
              if ( v12 )
              {
                v13 = sub_BD2DA0(136);
                v14 = (unsigned __int8 *)v13;
                if ( v13 )
                  sub_B2C3B0(v13, v10, 0, 0xFFFFFFFF, (__int64)v17, v11);
              }
              else
              {
                BYTE4(v16) = 0;
                v14 = (unsigned __int8 *)sub_BD2C40(88, unk_3F0FAE8);
                if ( v14 )
                  sub_B30000((__int64)v14, v11, (_QWORD *)v10, 0, 0, 0, (__int64)v17, 0, 0, v16, 0);
              }
              sub_BD6B90(v14, (unsigned __int8 *)a1);
              sub_BD84D0(a1, (__int64)v14);
              LOBYTE(v3) = sub_B30340((_QWORD *)a1);
            }
          }
          else
          {
            sub_B2CA40(a1, 0);
            LOBYTE(v3) = *(_BYTE *)(a1 + 32);
            *(_BYTE *)(a1 + 32) = v3 & 0xF0;
            if ( (v3 & 0x30) != 0 )
              *(_BYTE *)(a1 + 33) |= 0x40u;
          }
        }
        else
        {
          LOBYTE(v3) = sub_B30810((_QWORD *)a1);
        }
      }
      else
      {
        v9 = 1;
        while ( v8 != -4096 )
        {
          v7 = v6 & (v9 + v7);
          v8 = *(_QWORD *)(v5 + 8LL * v7);
          if ( v3 == v8 )
            goto LABEL_4;
          ++v9;
        }
      }
    }
  }
  return v3;
}
