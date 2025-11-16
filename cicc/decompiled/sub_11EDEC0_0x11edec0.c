// Function: sub_11EDEC0
// Address: 0x11edec0
//
__int64 __fastcall sub_11EDEC0(__int64 **a1, __int64 a2, __int64 *a3)
{
  __int64 v5; // r10
  __int64 v6; // r15
  __int64 v7; // rcx
  __int64 v8; // rsi
  __int64 v9; // r13
  __int64 v10; // rdx
  _QWORD *v11; // rbx
  _QWORD *v12; // r12
  __int64 v13; // rdi
  char v15; // [rsp+Fh] [rbp-D1h]
  __int64 v16; // [rsp+10h] [rbp-D0h]
  __int64 v17; // [rsp+18h] [rbp-C8h]
  int v18; // [rsp+2Ch] [rbp-B4h] BYREF
  _BYTE *v19; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v20; // [rsp+38h] [rbp-A8h]
  _BYTE v21[160]; // [rsp+40h] [rbp-A0h] BYREF

  v5 = *(_QWORD *)(a2 - 32);
  if ( v5 )
  {
    if ( *(_BYTE *)v5 )
    {
      v5 = 0;
    }
    else if ( *(_QWORD *)(v5 + 24) != *(_QWORD *)(a2 + 80) )
    {
      v5 = 0;
    }
  }
  v16 = v5;
  v15 = sub_97F320(a2);
  v20 = 0x200000000LL;
  v19 = v21;
  sub_B56970(a2, (__int64)&v19);
  v6 = a3[14];
  v17 = a3[15];
  v7 = (unsigned int)v20;
  a3[14] = (__int64)v19;
  v8 = v16;
  a3[15] = v7;
  if ( sub_981210(**a1, v16, (unsigned int *)&v18) && (v18 == 158 || v18 == 327 || v18 == 468 || v18 == 332 || v15) )
  {
    switch ( v18 )
    {
      case 120:
        v8 = a2;
        v9 = sub_11ED3A0(a1, a2, (__int64)a3);
        break;
      case 121:
        v8 = a2;
        v9 = sub_11ECB60((__int64)a1, a2, (__int64)a3);
        break;
      case 122:
        v8 = a2;
        v9 = sub_11ECC30((__int64)a1, a2, (__int64)a3);
        break;
      case 123:
        v8 = a2;
        v9 = (__int64)sub_11ECE30(a1, a2, (__int64)a3);
        break;
      case 124:
        v8 = a2;
        v9 = sub_11ECD00((__int64)a1, a2, a3);
        break;
      case 139:
        v8 = a2;
        v9 = sub_11ED480(a1, (unsigned __int8 *)a2, (__int64)a3);
        break;
      case 140:
        v8 = a2;
        v9 = sub_11ED760(a1, (unsigned __int8 *)a2, (__int64)a3);
        break;
      case 144:
      case 147:
        v8 = a2;
        v9 = sub_11ECEF0((__int64)a1, a2, (__int64)a3, v18);
        break;
      case 145:
      case 153:
        v8 = a2;
        v9 = sub_11ED2A0(a1, a2, (__int64)a3, v18);
        break;
      case 146:
        v8 = a2;
        v9 = sub_11EDA20(a1, a2, (__int64)a3);
        break;
      case 149:
        v8 = a2;
        v9 = sub_11EDAD0(a1, a2, (__int64)a3);
        break;
      case 150:
        v8 = a2;
        v9 = sub_11EDC50(a1, a2, (__int64)a3);
        break;
      case 151:
        v8 = a2;
        v9 = sub_11ED200(a1, a2, (__int64)a3);
        break;
      case 152:
        v8 = a2;
        v9 = sub_11EDB90(a1, a2, (__int64)a3);
        break;
      case 156:
        v8 = a2;
        v9 = sub_11EDD10(a1, a2, (__int64)a3);
        break;
      case 157:
        v8 = a2;
        v9 = sub_11EDDF0(a1, a2, (__int64)a3);
        break;
      default:
        goto LABEL_11;
    }
  }
  else
  {
LABEL_11:
    v9 = 0;
  }
  v10 = (unsigned int)v20;
  a3[14] = v6;
  a3[15] = v17;
  v11 = v19;
  v12 = &v19[56 * v10];
  if ( v19 != (_BYTE *)v12 )
  {
    do
    {
      v13 = *(v12 - 3);
      v12 -= 7;
      if ( v13 )
      {
        v8 = v12[6] - v13;
        j_j___libc_free_0(v13, v8);
      }
      if ( (_QWORD *)*v12 != v12 + 2 )
      {
        v8 = v12[2] + 1LL;
        j_j___libc_free_0(*v12, v8);
      }
    }
    while ( v11 != v12 );
    v12 = v19;
  }
  if ( v12 != (_QWORD *)v21 )
    _libc_free(v12, v8);
  return v9;
}
