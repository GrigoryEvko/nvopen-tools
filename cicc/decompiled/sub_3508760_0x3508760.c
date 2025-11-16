// Function: sub_3508760
// Address: 0x3508760
//
_QWORD *__fastcall sub_3508760(_QWORD *a1, __int64 *a2, unsigned __int8 (__fastcall *a3)(__int64))
{
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // r13
  __int64 v6; // r14
  __int64 v8; // [rsp+8h] [rbp-58h]
  __int64 v9; // [rsp+10h] [rbp-50h]
  __int64 v10; // [rsp+18h] [rbp-48h]
  __int64 v12; // [rsp+28h] [rbp-38h]

  v3 = *a2;
  v4 = a2[1];
  v5 = a2[2];
  v6 = a2[3];
  v10 = a2[6];
  v12 = a2[4];
  v8 = a2[5];
  v9 = a2[7];
  while ( 1 )
  {
LABEL_2:
    if ( v12 != v3 )
      goto LABEL_3;
LABEL_10:
    if ( v10 == v5 || v10 == v9 && v5 == v6 )
      break;
LABEL_3:
    if ( a3(v5) )
      break;
    v5 += 40;
    if ( v6 == v5 )
    {
      while ( 1 )
      {
        v3 = *(_QWORD *)(v3 + 8);
        if ( v4 == v3 )
          break;
        if ( (*(_BYTE *)(v3 + 44) & 4) == 0 )
        {
          v3 = v4;
          if ( v12 != v4 )
            goto LABEL_3;
          goto LABEL_10;
        }
        v5 = *(_QWORD *)(v3 + 32);
        v6 = v5 + 40LL * (*(_DWORD *)(v3 + 40) & 0xFFFFFF);
        if ( v5 != v6 )
          goto LABEL_2;
      }
    }
  }
  *a1 = v3;
  a1[1] = v4;
  a1[2] = v5;
  a1[3] = v6;
  a1[4] = v12;
  a1[9] = v12;
  a1[13] = v12;
  a1[5] = v8;
  a1[6] = v10;
  a1[7] = v9;
  a1[8] = a3;
  a1[10] = v8;
  a1[11] = v10;
  a1[12] = v9;
  a1[14] = v8;
  a1[15] = v10;
  a1[16] = v9;
  a1[17] = a3;
  return a1;
}
