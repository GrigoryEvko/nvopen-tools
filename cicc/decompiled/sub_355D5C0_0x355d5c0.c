// Function: sub_355D5C0
// Address: 0x355d5c0
//
__int64 *__fastcall sub_355D5C0(__int64 *a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  unsigned __int64 *v5; // rbx
  unsigned __int64 *v6; // r15
  __int64 v7; // r9
  unsigned __int64 *v8; // rdx
  unsigned __int64 v9; // rsi
  unsigned __int64 *v10; // rax
  __int64 **v11; // rbx
  __int64 **v12; // r15
  __int64 **v13; // r9
  __int64 ***v14; // r10
  __int64 ***v18; // [rsp+10h] [rbp-90h]
  __int64 v19; // [rsp+10h] [rbp-90h]
  __int64 **v20; // [rsp+18h] [rbp-88h]
  unsigned __int64 *v21; // [rsp+18h] [rbp-88h]
  unsigned __int64 v22[16]; // [rsp+20h] [rbp-80h] BYREF

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  a1[3] = 0;
  a1[4] = 0;
  a1[5] = 0;
  a1[6] = 0;
  a1[7] = 0;
  a1[8] = 0;
  a1[9] = 0;
  sub_3547BF0(a1, 0);
  v5 = (unsigned __int64 *)a4[2];
  v6 = (unsigned __int64 *)a4[4];
  v7 = a4[5];
  v8 = (unsigned __int64 *)a4[6];
  while ( v8 != v5 )
  {
    while ( 1 )
    {
      v9 = *v5;
      v22[0] = v9;
      if ( *(_WORD *)(*(_QWORD *)v9 + 68LL) == 68 || !*(_WORD *)(*(_QWORD *)v9 + 68LL) )
      {
        v10 = (unsigned __int64 *)a1[6];
        if ( v10 == (unsigned __int64 *)(a1[8] - 8) )
        {
          v19 = v7;
          v21 = v8;
          sub_354B0D0((unsigned __int64 *)a1, v22);
          v7 = v19;
          v8 = v21;
        }
        else
        {
          if ( v10 )
          {
            *v10 = v9;
            v10 = (unsigned __int64 *)a1[6];
          }
          a1[6] = (__int64)(v10 + 1);
        }
      }
      if ( v6 != ++v5 )
        break;
      v5 = *(unsigned __int64 **)(v7 + 8);
      v7 += 8;
      v6 = v5 + 64;
      if ( v8 == v5 )
        goto LABEL_11;
    }
  }
LABEL_11:
  memset(v22, 0, 80);
  sub_3547BF0((__int64 *)v22, 0);
  v11 = (__int64 **)a4[2];
  v12 = (__int64 **)a4[4];
  v13 = (__int64 **)a4[6];
  v14 = (__int64 ***)(a4[5] + 8LL);
  while ( v11 != v13 )
  {
    while ( 1 )
    {
      if ( *(_WORD *)(**v11 + 68) != 68 && *(_WORD *)(**v11 + 68) )
      {
        v18 = v14;
        v20 = v13;
        sub_355B330(a2, a3, *v11, v22);
        v14 = v18;
        v13 = v20;
      }
      if ( v12 != ++v11 )
        break;
      v11 = *v14++;
      v12 = v11 + 64;
      if ( v11 == v13 )
        goto LABEL_18;
    }
  }
LABEL_18:
  sub_355D240((unsigned __int64 *)a1, v22);
  sub_3546C50(v22);
  return a1;
}
