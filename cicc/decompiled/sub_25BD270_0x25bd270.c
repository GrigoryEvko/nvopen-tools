// Function: sub_25BD270
// Address: 0x25bd270
//
__int64 __fastcall sub_25BD270(int *a1, unsigned __int8 *a2, char a3, __int64 a4)
{
  int v6; // edx
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r15
  int v11; // r15d
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int8 *v14; // r15
  __m128i *v15; // r9
  __int64 result; // rax
  unsigned __int8 **v17; // rbx
  __int64 v18; // rcx
  int v19; // edx
  __m128i v20; // xmm0
  __m128i v21; // xmm1
  unsigned __int8 *v22; // [rsp+10h] [rbp-90h]
  __m128i *v23; // [rsp+18h] [rbp-88h]
  __m128i v24; // [rsp+20h] [rbp-80h] BYREF
  __m128i v25; // [rsp+30h] [rbp-70h] BYREF
  unsigned __int8 *v26[2]; // [rsp+40h] [rbp-60h] BYREF
  __m128i v27; // [rsp+50h] [rbp-50h]
  __m128i v28; // [rsp+60h] [rbp-40h]

  v6 = *a2;
  if ( v6 == 40 )
  {
    v7 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v7 = -32;
    if ( v6 != 85 )
    {
      v7 = -96;
      if ( v6 != 34 )
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) != 0 )
  {
    v8 = sub_BD2BC0((__int64)a2);
    v10 = v8 + v9;
    if ( (a2[7] & 0x80u) == 0 )
    {
      if ( !(unsigned int)(v10 >> 4) )
        goto LABEL_9;
    }
    else
    {
      if ( !(unsigned int)((v10 - sub_BD2BC0((__int64)a2)) >> 4) )
        goto LABEL_9;
      if ( (a2[7] & 0x80u) != 0 )
      {
        v11 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
        if ( (a2[7] & 0x80u) == 0 )
          BUG();
        v12 = sub_BD2BC0((__int64)a2);
        v7 -= 32LL * (unsigned int)(*(_DWORD *)(v12 + v13 - 4) - v11);
        goto LABEL_9;
      }
    }
    BUG();
  }
LABEL_9:
  v14 = &a2[v7];
  v15 = &v24;
  result = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
  v17 = (unsigned __int8 **)&a2[-result];
  if ( &a2[-result] != v14 )
  {
    do
    {
      v18 = *((_QWORD *)*v17 + 1);
      v19 = *(unsigned __int8 *)(v18 + 8);
      result = (unsigned int)(v19 - 17);
      if ( (unsigned int)result <= 1 )
      {
        result = **(_QWORD **)(v18 + 16);
        LOBYTE(v19) = *(_BYTE *)(result + 8);
      }
      if ( (_BYTE)v19 == 14 )
      {
        v23 = v15;
        v22 = *v17;
        sub_B91FC0(v15->m128i_i64, (__int64)a2);
        v20 = _mm_loadu_si128(&v24);
        v21 = _mm_loadu_si128(&v25);
        v26[1] = (unsigned __int8 *)-1LL;
        v27 = v20;
        v26[0] = v22;
        v28 = v21;
        result = sub_25BCA40(a1, v26, a3, a4);
        v15 = v23;
      }
      v17 += 4;
    }
    while ( v14 != (unsigned __int8 *)v17 );
  }
  return result;
}
