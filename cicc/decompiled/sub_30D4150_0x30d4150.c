// Function: sub_30D4150
// Address: 0x30d4150
//
__int64 __fastcall sub_30D4150(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  int v5; // eax
  __int64 v6; // rcx
  __int64 v7; // rsi
  int v8; // edi
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r8
  __int64 v12; // r13
  unsigned __int64 v13; // r14
  unsigned __int64 v14; // rax
  unsigned int v15; // r15d
  unsigned __int64 v16; // r12
  int v17; // eax
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // rcx
  unsigned int v20; // eax
  __int64 v21; // rax
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rdx
  int v24; // eax
  unsigned __int64 v26; // r13
  unsigned __int64 v27; // rcx
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // rcx
  int v31; // r9d

  v4 = sub_30D1740(a1, *(_QWORD *)(a2 - 32));
  if ( v4 )
    sub_30D1890(a1, v4);
  if ( (unsigned __int8)sub_B4CE70(a2) )
  {
    v5 = *(_DWORD *)(a1 + 160);
    v6 = *(_QWORD *)(a2 - 32);
    v7 = *(_QWORD *)(a1 + 144);
    if ( v5 )
    {
      v8 = v5 - 1;
      v9 = (v5 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v10 = (__int64 *)(v7 + 16LL * v9);
      v11 = *v10;
      if ( v6 == *v10 )
      {
LABEL_6:
        v12 = v10[1];
        if ( v12 && *(_BYTE *)v12 == 17 )
        {
          v13 = *(_QWORD *)(a1 + 120);
          v14 = sub_BDB740(*(_QWORD *)(a1 + 80), *(_QWORD *)(a2 + 72));
          v15 = *(_DWORD *)(v12 + 32);
          v16 = v14;
          if ( v15 <= 0x40 )
          {
            v18 = *(_QWORD *)(v12 + 24);
          }
          else
          {
            if ( v15 - (unsigned int)sub_C444A0(v12 + 24) > 0x40 )
            {
              v17 = 63;
              v18 = -1;
              goto LABEL_11;
            }
            v18 = **(_QWORD **)(v12 + 24);
          }
          if ( !v18 )
            goto LABEL_33;
          _BitScanReverse64(&v30, v18);
          v17 = 63 - (v30 ^ 0x3F);
LABEL_11:
          if ( v16 )
          {
            _BitScanReverse64(&v19, v16);
            v20 = v17 - (v19 ^ 0x3F);
            if ( v20 < 0xFFFFFFC1 )
            {
              if ( v20 )
                goto LABEL_43;
              v21 = v16 * (v18 >> 1);
              if ( v21 < 0 )
                goto LABEL_43;
              v22 = 2 * v21;
              if ( (v18 & 1) == 0 )
                goto LABEL_34;
              v23 = v16 + v22;
              if ( v16 >= v22 )
                v22 = v16;
              if ( v23 < v22 )
              {
LABEL_43:
                *(_QWORD *)(a1 + 120) = -1;
LABEL_24:
                *(_BYTE *)(a1 + 107) = 1;
                return 0;
              }
              v22 = v23;
LABEL_34:
              v29 = v13 + v22;
              if ( v13 >= v22 )
                v22 = v13;
              if ( v29 >= v22 )
              {
                *(_QWORD *)(a1 + 120) = v29;
                if ( v29 <= 0x10000 )
                  return 0;
                goto LABEL_24;
              }
              goto LABEL_43;
            }
          }
LABEL_33:
          v22 = v18 * v16;
          goto LABEL_34;
        }
      }
      else
      {
        v24 = 1;
        while ( v11 != -4096 )
        {
          v31 = v24 + 1;
          v9 = v8 & (v24 + v9);
          v10 = (__int64 *)(v7 + 16LL * v9);
          v11 = *v10;
          if ( v6 == *v10 )
            goto LABEL_6;
          v24 = v31;
        }
      }
    }
  }
  if ( sub_B4D040(a2) )
  {
    v26 = *(_QWORD *)(a1 + 120);
    v27 = sub_BDB740(*(_QWORD *)(a1 + 80), *(_QWORD *)(a2 + 72));
    v28 = v26 + v27;
    if ( v26 < v27 )
      v26 = v27;
    if ( v28 < v26 )
      v28 = -1;
    *(_QWORD *)(a1 + 120) = v28;
  }
  if ( !sub_B4D040(a2) )
    goto LABEL_24;
  return 0;
}
