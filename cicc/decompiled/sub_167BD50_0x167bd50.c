// Function: sub_167BD50
// Address: 0x167bd50
//
char __fastcall sub_167BD50(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // rax
  int v12; // edx
  int v13; // ecx
  __int64 v14; // rdi
  int v15; // r8d
  unsigned int v16; // edx
  __int64 v17; // rsi
  char v18; // al
  __int64 v19; // r14
  __int64 v20; // r15
  __int64 v21; // rax
  _QWORD *v22; // r13
  double v23; // xmm4_8
  double v24; // xmm5_8
  __int64 v25; // r14
  _BYTE v27[16]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v28; // [rsp+10h] [rbp-40h]

  v11 = sub_15E4F10(a1);
  if ( v11 )
  {
    v12 = *(_DWORD *)(a2 + 24);
    if ( v12 )
    {
      v13 = v12 - 1;
      v14 = *(_QWORD *)(a2 + 8);
      v15 = 1;
      v16 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v17 = *(_QWORD *)(v14 + 8LL * v16);
      if ( v11 == v17 )
      {
LABEL_4:
        if ( *(_QWORD *)(a1 + 8) )
        {
          v18 = *(_BYTE *)(a1 + 16);
          if ( v18 )
          {
            if ( v18 == 3 )
            {
              LOBYTE(v11) = sub_15E5440(a1, 0);
            }
            else
            {
              v19 = *(_QWORD *)(a1 + 24);
              v20 = *(_QWORD *)(a1 + 40);
              if ( *(_BYTE *)(v19 + 8) == 12 )
              {
                v28 = 257;
                v21 = sub_1648B60(120);
                v22 = (_QWORD *)v21;
                if ( v21 )
                  sub_15E2490(v21, v19, 0, (__int64)v27, v20);
              }
              else
              {
                v25 = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
                v28 = 257;
                v22 = sub_1648A60(88, 1u);
                if ( v22 )
                  sub_15E51E0((__int64)v22, v20, v25, 0, 0, 0, (__int64)v27, 0, 0, 0, 0);
              }
              sub_164B7C0((__int64)v22, a1);
              sub_164D160(a1, (__int64)v22, a3, a4, a5, a6, v23, v24, a9, a10);
              LOBYTE(v11) = sub_15E58C0((_QWORD *)a1);
            }
          }
          else
          {
            sub_15E0C30(a1);
            LOBYTE(v11) = *(_BYTE *)(a1 + 32);
            *(_BYTE *)(a1 + 32) = v11 & 0xF0;
            if ( (v11 & 0x30) != 0 )
              *(_BYTE *)(a1 + 33) |= 0x40u;
          }
        }
        else
        {
          LOBYTE(v11) = sub_15E5B20(a1);
        }
      }
      else
      {
        while ( v17 != -8 )
        {
          v16 = v13 & (v15 + v16);
          v17 = *(_QWORD *)(v14 + 8LL * v16);
          if ( v11 == v17 )
            goto LABEL_4;
          ++v15;
        }
      }
    }
  }
  return v11;
}
