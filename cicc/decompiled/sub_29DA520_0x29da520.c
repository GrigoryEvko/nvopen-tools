// Function: sub_29DA520
// Address: 0x29da520
//
__int64 __fastcall sub_29DA520(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  unsigned int v5; // r14d
  __int64 v6; // rax
  unsigned int v7; // r15d
  __int64 v9; // r15
  unsigned int v10; // eax
  __int64 v11; // r14
  __int64 v12; // rax
  unsigned int v13; // eax
  __int64 v14; // r14
  unsigned int v15; // eax
  __int64 v16; // [rsp+8h] [rbp-58h]
  unsigned int v17; // [rsp+8h] [rbp-58h]
  unsigned __int64 v18; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-48h]
  unsigned __int64 v20; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v21; // [rsp+28h] [rbp-38h]

  v4 = *(_QWORD *)(*(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)) + 8LL);
  if ( (unsigned int)*(unsigned __int8 *)(v4 + 8) - 17 <= 1 )
    v4 = **(_QWORD **)(v4 + 16);
  v5 = *(_DWORD *)(v4 + 8) >> 8;
  v6 = *(_QWORD *)(*(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)) + 8LL);
  if ( (unsigned int)*(unsigned __int8 *)(v6 + 8) - 17 <= 1 )
    v6 = **(_QWORD **)(v6 + 16);
  v7 = sub_29D7CF0((__int64)a1, v5, *(_DWORD *)(v6 + 8) >> 8);
  if ( !v7 )
  {
    v9 = sub_B2BEC0(*a1);
    v10 = sub_AE2980(v9, v5)[3];
    v19 = v10;
    if ( v10 > 0x40 )
    {
      v17 = v10;
      sub_C43690((__int64)&v18, 0, 0);
      v21 = v17;
      sub_C43690((__int64)&v20, 0, 0);
    }
    else
    {
      v18 = 0;
      v21 = v10;
      v20 = 0;
    }
    if ( (unsigned __int8)sub_BB6360(a2, v9, (__int64)&v18, 0, 0)
      && (unsigned __int8)sub_BB6360(a3, v9, (__int64)&v20, 0, 0) )
    {
      v7 = sub_29D7D50((__int64)a1, (__int64)&v18, (__int64)&v20);
    }
    else
    {
      v11 = sub_BB5290(a3);
      v12 = sub_BB5290(a2);
      v7 = sub_29D81B0(a1, v12, v11);
      if ( !v7 )
      {
        v7 = sub_29D7CF0((__int64)a1, *(_DWORD *)(a2 + 4) & 0x7FFFFFF, *(_DWORD *)(a3 + 4) & 0x7FFFFFF);
        if ( !v7 )
        {
          v13 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
          if ( v13 )
          {
            v14 = 0;
            v16 = v13 - 1;
            while ( 1 )
            {
              v15 = sub_29DA390(
                      (__int64)a1,
                      *(_QWORD *)(a2 + 32 * (v14 - v13)),
                      *(_QWORD *)(a3 + 32 * (v14 - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF))));
              if ( v15 )
                break;
              if ( v14 == v16 )
                goto LABEL_12;
              ++v14;
              v13 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
            }
            v7 = v15;
          }
        }
      }
    }
LABEL_12:
    if ( v21 > 0x40 && v20 )
      j_j___libc_free_0_0(v20);
    if ( v19 > 0x40 && v18 )
      j_j___libc_free_0_0(v18);
  }
  return v7;
}
