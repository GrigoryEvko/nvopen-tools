// Function: sub_31CDA70
// Address: 0x31cda70
//
__int64 __fastcall sub_31CDA70(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 *v5; // rax
  __int64 v6; // r15
  __int64 *v7; // r14
  __int64 v8; // rbx
  __int64 v9; // rdx
  unsigned int v10; // esi
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // r14
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rcx
  _QWORD *v17; // r13
  unsigned int v18; // [rsp+8h] [rbp-88h]
  __int64 v19[4]; // [rsp+10h] [rbp-80h] BYREF
  _QWORD v20[4]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v21; // [rsp+50h] [rbp-40h]

  result = sub_BD2910(*(_QWORD *)(a1 + 312));
  if ( !(_DWORD)result )
  {
    result = *(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL);
    if ( (unsigned int)*(unsigned __int8 *)(result + 8) - 17 <= 1 )
      result = **(_QWORD **)(result + 16);
    if ( *(_DWORD *)(result + 8) <= 0x1FFu )
    {
      v5 = (__int64 *)sub_B43CA0(a2);
      v6 = *(_QWORD *)(a2 + 8);
      v7 = v5;
      v18 = (*(_WORD *)(a2 + 2) >> 1) & 7 | (((*(_WORD *)(a2 + 2) >> 4) & 0x1F) << 16);
      v8 = sub_BCB2D0((_QWORD *)*v5);
      v9 = sub_BCB2E0((_QWORD *)*v7);
      result = *(unsigned __int8 *)(v6 + 8);
      switch ( (_BYTE)result )
      {
        case 3:
          v10 = 8916;
          break;
        case 0xC:
          if ( v9 == v6 )
          {
            v10 = 8918;
          }
          else
          {
            v10 = 8917;
            if ( v8 != v6 )
              return result;
          }
          break;
        case 2:
          v10 = 8915;
          break;
        default:
          return result;
      }
      v20[0] = *(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL);
      v11 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
      v20[2] = v9;
      v20[1] = v11;
      v12 = v8;
      v13 = sub_B6E160(v7, v10, (__int64)v20, 3);
      v14 = 0;
      v19[0] = sub_AD64C0(v12, v18, 0);
      v15 = *(_QWORD *)(a2 - 64);
      v21 = 257;
      v16 = *(_QWORD *)(a1 + 304);
      v19[1] = v15;
      v19[2] = *(_QWORD *)(a2 - 32);
      v19[3] = *(_QWORD *)(v16 + 32 * (1LL - (*(_DWORD *)(v16 + 4) & 0x7FFFFFF)));
      if ( v13 )
        v14 = *(_QWORD *)(v13 + 24);
      v17 = sub_BD2C40(88, 5u);
      if ( v17 )
      {
        sub_B44260((__int64)v17, **(_QWORD **)(v14 + 16), 56, 5u, a2 + 24, 0);
        v17[9] = 0;
        sub_B4A290((__int64)v17, v14, v13, v19, 4, (__int64)v20, 0, 0);
      }
      sub_BD84D0(a2, (__int64)v17);
      return sub_B43D60((_QWORD *)a2);
    }
  }
  return result;
}
