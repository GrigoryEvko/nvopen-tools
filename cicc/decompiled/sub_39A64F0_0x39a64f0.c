// Function: sub_39A64F0
// Address: 0x39a64f0
//
__int64 __fastcall sub_39A64F0(__int64 *a1, __int64 a2)
{
  __int64 v3; // r12
  __int16 v4; // ax
  unsigned __int8 *v5; // rbx
  __int64 v6; // r15
  __int64 v7; // r14
  unsigned __int8 v8; // al
  __int64 v10; // rcx
  __int64 v11; // r8
  int v12; // r9d
  __m128i *v13; // r15
  __int64 v14; // rdi
  size_t v15; // rdx
  __int64 v16; // rax
  __int16 v17; // [rsp+0h] [rbp-40h]
  int *v18; // [rsp+0h] [rbp-40h]
  size_t v19; // [rsp+8h] [rbp-38h]

  if ( !a2 )
    return 0;
  v3 = a2;
  while ( 1 )
  {
    v4 = *(_WORD *)(v3 + 2);
    if ( v4 != 55 )
    {
      if ( v4 != 71 )
        break;
      goto LABEL_4;
    }
    if ( (unsigned __int16)sub_398C0A0(a1[25]) <= 2u )
      goto LABEL_5;
    if ( *(_WORD *)(v3 + 2) != 71 )
      break;
LABEL_4:
    if ( (unsigned __int16)sub_398C0A0(a1[25]) > 4u )
      break;
LABEL_5:
    v3 = *(_QWORD *)(v3 + 8 * (3LL - *(unsigned int *)(v3 + 8)));
    if ( !v3 )
      return 0;
  }
  v5 = *(unsigned __int8 **)(v3 + 8 * (1LL - *(unsigned int *)(v3 + 8)));
  v6 = sub_39A81B0(a1);
  v7 = (__int64)sub_39A23D0((__int64)a1, (unsigned __int8 *)v3);
  if ( !v7 )
  {
    if ( v6 )
    {
      v7 = sub_39A5A90((__int64)a1, *(_WORD *)(v3 + 2), v6, (unsigned __int8 *)v3);
    }
    else
    {
      v17 = *(_WORD *)(v3 + 2);
      v7 = sub_145CBF0(a1 + 11, 48, 16);
      *(_QWORD *)v7 = v7 | 4;
      *(_QWORD *)(v7 + 8) = 0;
      *(_QWORD *)(v7 + 16) = 0;
      *(_DWORD *)(v7 + 24) = -1;
      *(_WORD *)(v7 + 28) = v17;
      *(_BYTE *)(v7 + 30) = 0;
      *(_QWORD *)(v7 + 32) = 0;
      *(_QWORD *)(v7 + 40) = 0;
      sub_39A55B0((__int64)a1, (unsigned __int8 *)v3, (unsigned __int8 *)v7);
      sub_39A6490((__int64)a1, (__int64)v5, v7, v10, v11, v12);
    }
    sub_39A29E0(a1, v5, v3, v7);
    v8 = *(_BYTE *)v3;
    if ( *(_BYTE *)v3 == 11 )
    {
      sub_39A4140(a1, v7, v3);
    }
    else
    {
      switch ( v8 )
      {
        case 0x20u:
          sub_39A41F0(a1, v7, v3);
          break;
        case 0xEu:
          sub_39A6910(a1, v7, v3);
          break;
        case 0x21u:
          sub_39A7A30(a1, v7, v3);
          break;
        case 0xDu:
          v13 = (__m128i *)a1[25];
          if ( v13[281].m128i_i8[8]
            && (*(_BYTE *)(v3 + 28) & 4) == 0
            && (v14 = *(_QWORD *)(v3 + 8 * (7LL - *(unsigned int *)(v3 + 8)))) != 0 )
          {
            v18 = (int *)sub_161E970(v14);
            v19 = v15;
            v16 = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 40))(a1);
            sub_3996FA0(v13, v16, v18, v19, v7, v3);
          }
          else
          {
            sub_39A8AE0(a1, v7, v3);
          }
          break;
        default:
          sub_39A88F0(a1, v7, v3);
          break;
      }
    }
  }
  return v7;
}
