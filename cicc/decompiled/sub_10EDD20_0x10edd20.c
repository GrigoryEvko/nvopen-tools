// Function: sub_10EDD20
// Address: 0x10edd20
//
__int64 __fastcall sub_10EDD20(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r14
  __int64 v4; // r12
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // r13
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v17; // rax
  __int64 v18; // r14
  __int64 v19; // rcx
  unsigned __int64 v20; // rdx
  char v21; // bl
  __int64 v22; // r15
  _QWORD *v23; // rax
  __int64 v24; // r9
  __int64 v25; // r13
  __int64 v26; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v27; // [rsp+8h] [rbp-58h]
  __int64 v28; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v29; // [rsp+18h] [rbp-48h]
  __int64 v30; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v31; // [rsp+28h] [rbp-38h]

  v2 = 32 * (3LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v3 = *(_QWORD *)(a2 + v2);
  if ( *(_BYTE *)v3 <= 0x15u )
  {
    v4 = a2;
    if ( sub_AC30F0(*(_QWORD *)(a2 + v2)) )
      return sub_F207A0(a1, (__int64 *)a2);
    if ( sub_AD7930((_BYTE *)v3, a2, v5, v6, v7) )
    {
      v17 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
      v18 = *(_QWORD *)(a2 + 32 * (1 - v17));
      v19 = *(_QWORD *)(a2 + 32 * (2 - v17));
      v20 = *(_QWORD *)(v19 + 24);
      if ( *(_DWORD *)(v19 + 32) > 0x40u )
        v20 = *(_QWORD *)v20;
      v21 = 0;
      if ( v20 )
      {
        _BitScanReverse64(&v20, v20);
        v21 = 63 - (v20 ^ 0x3F);
      }
      v22 = *(_QWORD *)(a2 - 32 * v17);
      v23 = sub_BD2C40(80, unk_3F10A10);
      v25 = (__int64)v23;
      if ( v23 )
        sub_B4D3C0((__int64)v23, v22, v18, 0, v21, v24, 0, 0);
      sub_B47C00(v25, a2, 0, 0);
      return v25;
    }
    if ( *(_BYTE *)(*(_QWORD *)(v3 + 8) + 8LL) != 18 )
    {
      sub_9BA1B0(&v26, v3);
      v29 = v27;
      if ( v27 > 0x40 )
      {
        sub_C43690((__int64)&v28, 0, 0);
        v31 = v27;
        if ( v27 > 0x40 )
        {
          sub_C43780((__int64)&v30, (const void **)&v26);
LABEL_8:
          v8 = sub_11A3F30(a1, *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), &v30, &v28, 0, 0);
          if ( v31 > 0x40 && v30 )
            j_j___libc_free_0_0(v30);
          if ( v8 )
          {
            if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
              v9 = *(_QWORD *)(a2 - 8);
            else
              v9 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
            v10 = *(_QWORD *)v9;
            if ( *(_QWORD *)v9 )
            {
              v11 = *(_QWORD *)(v9 + 8);
              **(_QWORD **)(v9 + 16) = v11;
              if ( v11 )
                *(_QWORD *)(v11 + 16) = *(_QWORD *)(v9 + 16);
            }
            *(_QWORD *)v9 = v8;
            v12 = *(_QWORD *)(v8 + 16);
            *(_QWORD *)(v9 + 8) = v12;
            if ( v12 )
              *(_QWORD *)(v12 + 16) = v9 + 8;
            *(_QWORD *)(v9 + 16) = v8 + 16;
            *(_QWORD *)(v8 + 16) = v9;
            if ( *(_BYTE *)v10 > 0x1Cu )
            {
              v13 = *(_QWORD *)(a1 + 40);
              v30 = v10;
              v14 = v13 + 2096;
              sub_10E8740(v14, &v30);
              v15 = *(_QWORD *)(v10 + 16);
              if ( v15 )
              {
                if ( !*(_QWORD *)(v15 + 8) )
                {
                  v30 = *(_QWORD *)(v15 + 24);
                  sub_10E8740(v14, &v30);
                }
              }
            }
          }
          else
          {
            v4 = 0;
          }
          if ( v29 > 0x40 && v28 )
            j_j___libc_free_0_0(v28);
          if ( v27 > 0x40 )
          {
            if ( v26 )
              j_j___libc_free_0_0(v26);
          }
          return v4;
        }
      }
      else
      {
        v28 = 0;
        v31 = v27;
      }
      v30 = v26;
      goto LABEL_8;
    }
  }
  return 0;
}
