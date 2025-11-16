// Function: sub_83CDB0
// Address: 0x83cdb0
//
__int64 __fastcall sub_83CDB0(__int64 a1, __m128i *a2, __int64 a3, __int64 *a4)
{
  __m128i *v5; // r13
  __int64 v6; // r15
  int v7; // ebx
  char v8; // al
  __int64 v9; // r12
  __int64 v10; // rdx
  unsigned int v11; // r15d
  __int64 v13; // rdx
  __int64 v14; // rax
  __m128i *v16; // [rsp+0h] [rbp-F0h]
  int v17; // [rsp+14h] [rbp-DCh]
  int v19; // [rsp+24h] [rbp-CCh] BYREF
  __m128i *v20; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v21; // [rsp+30h] [rbp-C0h] BYREF
  __m128i *v22; // [rsp+38h] [rbp-B8h] BYREF
  __m128i *v23; // [rsp+40h] [rbp-B0h] BYREF
  const __m128i *v24; // [rsp+48h] [rbp-A8h] BYREF
  __m128i v25; // [rsp+50h] [rbp-A0h] BYREF

  v5 = a2;
  if ( !(unsigned int)sub_8D3EA0(a2) )
  {
    v7 = 0;
    if ( (unsigned int)sub_8D32E0(a2) )
      v5 = (__m128i *)sub_8D46C0(a2);
    v17 = sub_8D3410(v5);
    if ( v17 )
    {
      while ( v5[8].m128i_i8[12] == 12 )
        v5 = (__m128i *)v5[10].m128i_i64[0];
      v16 = v5;
      v6 = *(_QWORD *)(a1 + 24);
      v5 = (__m128i *)v5[10].m128i_i64[0];
      v20 = v5;
      if ( !v6 )
      {
        v9 = 0;
        v11 = 1;
        goto LABEL_11;
      }
      v17 = 1;
      goto LABEL_4;
    }
    if ( (unsigned int)sub_828BC0((__int64)v5, &v20) )
    {
      v6 = *(_QWORD *)(a1 + 24);
      if ( v6 )
      {
        v16 = v5;
        v7 = 0;
        v5 = v20;
        goto LABEL_4;
      }
    }
    return 1;
  }
  v6 = *(_QWORD *)(a1 + 24);
  v20 = a2;
  if ( !v6 )
    return 1;
  v16 = a2;
  v7 = 1;
  v17 = 0;
LABEL_4:
  v8 = *(_BYTE *)(v6 + 8);
  v9 = 0;
  while ( 1 )
  {
    v24 = v5;
    ++v9;
    if ( v8 != 1 )
    {
      if ( v8 )
        goto LABEL_9;
      v10 = *(_QWORD *)(v6 + 24);
      v23 = *(__m128i **)(v10 + 8);
      if ( (unsigned int)sub_82C250(&v24, &v23, v10 + 8, a3, *a4, &v21, &v22, &v19) )
      {
        if ( (unsigned int)sub_8DBE70(v24) )
        {
          if ( !(unsigned int)sub_828690((__int64)v24, (__int64)v23, v21, (__int64)v22, (__int64)a4, a3) )
            goto LABEL_9;
        }
        else if ( v17 )
        {
          sub_839D30(v6, v20, 0, 0, 0, 0, 0, 0, 0, 0, 0, &v25);
          if ( v25.m128i_i32[2] == 7 )
          {
LABEL_9:
            v11 = 0;
            goto LABEL_10;
          }
        }
      }
      else if ( !v19 )
      {
        goto LABEL_9;
      }
LABEL_14:
      v13 = *(_QWORD *)v6;
      if ( !*(_QWORD *)v6 )
        break;
      goto LABEL_15;
    }
    if ( v7 )
      goto LABEL_14;
    if ( !(unsigned int)sub_83CDB0(v6, v5, a3, a4) )
      goto LABEL_9;
    v13 = *(_QWORD *)v6;
    if ( !*(_QWORD *)v6 )
      break;
LABEL_15:
    v8 = *(_BYTE *)(v13 + 8);
    if ( v8 == 3 )
    {
      v14 = sub_6BBB10((_QWORD *)v6);
      v13 = v14;
      if ( !v14 )
        break;
      v8 = *(_BYTE *)(v14 + 8);
    }
    v5 = v20;
    v6 = v13;
  }
  v11 = 1;
LABEL_10:
  if ( v17 )
  {
LABEL_11:
    if ( v16[10].m128i_i8[8] < 0 && !(unsigned int)sub_8B4E50(v9, v16[11].m128i_i64[0], a4, a3, 0) )
      return 0;
  }
  return v11;
}
