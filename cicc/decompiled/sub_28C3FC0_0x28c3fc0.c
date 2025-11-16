// Function: sub_28C3FC0
// Address: 0x28c3fc0
//
unsigned __int8 *__fastcall sub_28C3FC0(__int64 *a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // r15
  char v10; // al
  unsigned __int8 *result; // rax
  __int64 v12; // rcx
  __int64 ***v13; // r15
  __int64 ***v14; // [rsp+8h] [rbp-88h]
  __m128i v15; // [rsp+10h] [rbp-80h] BYREF
  __int64 v16; // [rsp+20h] [rbp-70h]
  __int64 v17; // [rsp+28h] [rbp-68h]
  __int64 v18; // [rsp+30h] [rbp-60h]
  __int64 v19; // [rsp+38h] [rbp-58h]
  __int64 v20; // [rsp+40h] [rbp-50h]
  __int64 v21; // [rsp+48h] [rbp-48h]
  __int16 v22; // [rsp+50h] [rbp-40h]

  v6 = a1[2];
  v7 = *a1;
  v19 = a2;
  v8 = a1[1];
  v17 = v6;
  LODWORD(v6) = *(_DWORD *)(a2 + 4);
  v18 = v7;
  v22 = 257;
  v15 = (__m128i)(unsigned __int64)v8;
  v16 = 0;
  v9 = *(_QWORD *)(a2 + 32 * (a3 + 1 - (unsigned __int64)(v6 & 0x7FFFFFF)));
  v20 = 0;
  v21 = 0;
  v10 = *(_BYTE *)v9;
  if ( *(_BYTE *)v9 <= 0x1Cu )
    goto LABEL_11;
  if ( v10 == 69 )
    goto LABEL_8;
  if ( v10 != 68 )
    goto LABEL_4;
  if ( (unsigned __int8)sub_9AC470(*(_QWORD *)(v9 - 32), &v15, 0) )
LABEL_8:
    v9 = *(_QWORD *)(v9 - 32);
  v10 = *(_BYTE *)v9;
  if ( *(_BYTE *)v9 <= 0x1Cu )
  {
LABEL_11:
    if ( v10 != 5 || *(_WORD *)(v9 + 2) != 13 )
      return 0;
    goto LABEL_12;
  }
LABEL_4:
  if ( v10 != 42 )
    return 0;
LABEL_12:
  if ( sub_28C1CA0((__int64)a1, v9, a2) && (unsigned int)sub_9B0030(v9, &v15) != 3 )
    return 0;
  v12 = *(_QWORD *)(v9 - 64);
  v13 = *(__int64 ****)(v9 - 32);
  v14 = (__int64 ***)v12;
  result = sub_28C3840(a1, a2, a3, v12, v13, a4);
  if ( !result )
  {
    if ( v13 != v14 )
      return sub_28C3840(a1, a2, a3, (__int64)v13, v14, a4);
    return 0;
  }
  return result;
}
