// Function: sub_28AAA80
// Address: 0x28aaa80
//
__int64 __fastcall sub_28AAA80(__int64 a1, __int64 *a2, unsigned __int8 *a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r12d
  __int64 v7; // rbx
  unsigned int v8; // r12d
  __int64 v10; // rdx
  __int64 v12; // r14
  __int64 v13; // rdi
  __int64 v14; // rdx
  _QWORD *v15; // rax
  _QWORD *v16; // rdx
  unsigned __int8 *v17; // r12
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned int v20; // ebx
  __int64 v21; // r13
  __int64 v22; // rax
  _QWORD v23[6]; // [rsp+0h] [rbp-80h] BYREF
  _QWORD v24[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v25; // [rsp+40h] [rbp-40h]
  __int64 v26; // [rsp+48h] [rbp-38h]
  __int64 v27; // [rsp+50h] [rbp-30h]
  __int64 v28; // [rsp+58h] [rbp-28h]

  if ( a4 != *(_QWORD *)(a1 + 128) )
  {
    v7 = *(_QWORD *)(a4 + 72);
    if ( v7 )
    {
      if ( *(_BYTE *)v7 == 85 )
      {
        v10 = *(_QWORD *)(v7 - 32);
        if ( v10 )
        {
          if ( !*(_BYTE *)v10
            && *(_QWORD *)(v10 + 24) == *(_QWORD *)(v7 + 80)
            && (*(_BYTE *)(v10 + 33) & 0x20) != 0
            && *(_DWORD *)(v10 + 36) == 211 )
          {
            v12 = *(_QWORD *)(v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF));
            if ( *(_BYTE *)a5 == 17 )
            {
              v13 = *a2;
              v14 = *(_QWORD *)(v7 + 32 * (1LL - (*(_DWORD *)(v7 + 4) & 0x7FFFFFF)));
              v24[1] = 1;
              v25 = 0;
              v24[0] = v14;
              v26 = 0;
              v27 = 0;
              v28 = 0;
              v23[0] = a3;
              v23[1] = 1;
              memset(&v23[2], 0, 32);
              if ( (unsigned __int8)sub_CF4D50(v13, (__int64)v23, (__int64)v24, (__int64)(a2 + 1), 0) == 3 )
              {
                v15 = *(_QWORD **)(v12 + 24);
                if ( *(_DWORD *)(v12 + 32) > 0x40u )
                  v15 = (_QWORD *)*v15;
                v16 = *(_QWORD **)(a5 + 24);
                if ( *(_DWORD *)(a5 + 32) > 0x40u )
                  v16 = (_QWORD *)*v16;
                v8 = 1;
                if ( v16 <= v15 )
                  return v8;
              }
            }
            v17 = sub_98ACB0(a3, 6u);
            if ( *v17 == 60
              && v17 == sub_98ACB0(*(unsigned __int8 **)(v7 + 32 * (1LL - (*(_DWORD *)(v7 + 4) & 0x7FFFFFF))), 6u) )
            {
              v18 = sub_B43CC0((__int64)v17);
              sub_B4CED0((__int64)v24, (__int64)v17, v18);
              v8 = (unsigned __int8)v25;
              if ( (_BYTE)v25 )
              {
                v19 = sub_CA1930(v24);
                v20 = *(_DWORD *)(v12 + 32);
                v21 = v19;
                if ( v20 <= 0x40 )
                {
                  v22 = *(_QWORD *)(v12 + 24);
                  goto LABEL_24;
                }
                if ( v20 - (unsigned int)sub_C444A0(v12 + 24) <= 0x40 )
                {
                  v22 = **(_QWORD **)(v12 + 24);
LABEL_24:
                  if ( v21 == v22 )
                    return v8;
                }
              }
            }
          }
        }
      }
    }
    return 0;
  }
  LOBYTE(v5) = *sub_98ACB0(a3, 6u) == 60;
  return v5;
}
