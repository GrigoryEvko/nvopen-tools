// Function: sub_213CCB0
// Address: 0x213ccb0
//
__int64 *__fastcall sub_213CCB0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rdx
  char v3; // al
  __int64 v4; // rdx
  unsigned int v5; // eax
  __int64 v6; // r15
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r9
  __int64 v11; // r8
  __int64 v12; // rdx
  __int64 *v13; // rdx
  __int64 v14; // rcx
  _BYTE *v15; // rdx
  __int64 *v16; // r12
  __int64 v18; // [rsp+0h] [rbp-170h]
  __int64 v19; // [rsp+8h] [rbp-168h]
  char v20[8]; // [rsp+20h] [rbp-150h] BYREF
  __int64 v21; // [rsp+28h] [rbp-148h]
  _BYTE *v22; // [rsp+30h] [rbp-140h] BYREF
  __int64 v23; // [rsp+38h] [rbp-138h]
  _BYTE v24[304]; // [rsp+40h] [rbp-130h] BYREF

  v2 = a2[5];
  v3 = *(_BYTE *)v2;
  v4 = *(_QWORD *)(v2 + 8);
  v20[0] = v3;
  v21 = v4;
  if ( v3 )
    v5 = word_4310720[(unsigned __int8)(v3 - 14)];
  else
    v5 = sub_1F58D30((__int64)v20);
  v22 = v24;
  v23 = 0x1000000000LL;
  if ( v5 )
  {
    v6 = 0;
    v7 = 40LL * v5;
    do
    {
      v8 = sub_2138AD0(a1, *(_QWORD *)(a2[4] + v6), *(_QWORD *)(a2[4] + v6 + 8));
      v10 = v9;
      v11 = v8;
      v12 = (unsigned int)v23;
      if ( (unsigned int)v23 >= HIDWORD(v23) )
      {
        v18 = v8;
        v19 = v10;
        sub_16CD150((__int64)&v22, v24, 0, 16, v8, v10);
        v12 = (unsigned int)v23;
        v11 = v18;
        v10 = v19;
      }
      v13 = (__int64 *)&v22[16 * v12];
      v6 += 40;
      *v13 = v11;
      v13[1] = v10;
      v14 = (unsigned int)(v23 + 1);
      LODWORD(v23) = v23 + 1;
    }
    while ( v7 != v6 );
    v15 = v22;
  }
  else
  {
    v14 = 0;
    v15 = v24;
  }
  v16 = sub_1D2E160(*(_QWORD **)(a1 + 8), a2, (__int64)v15, v14);
  if ( v22 != v24 )
    _libc_free((unsigned __int64)v22);
  return v16;
}
