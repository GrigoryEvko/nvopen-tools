// Function: sub_23B35A0
// Address: 0x23b35a0
//
void __fastcall sub_23B35A0(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // rbx
  _QWORD *i; // r14
  _QWORD *v4; // rsi
  __int64 v5; // rdx
  _BYTE *v6; // rax
  __int64 v7; // rdx
  _QWORD *v8; // [rsp+10h] [rbp-50h] BYREF
  __int64 v9; // [rsp+18h] [rbp-48h]
  _BYTE v10[64]; // [rsp+20h] [rbp-40h] BYREF

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0x2800000000LL;
  v2 = *(_QWORD **)(a2 + 112);
  for ( i = &v2[*(unsigned int *)(a2 + 120)]; i != v2; ++v2 )
  {
    v6 = (_BYTE *)sub_2E31BC0(*v2);
    v8 = v10;
    if ( v6 )
    {
      sub_23AE760((__int64 *)&v8, v6, (__int64)&v6[v7]);
      v4 = v8;
      v5 = v9;
    }
    else
    {
      v4 = v10;
      v9 = 0;
      v5 = 0;
      v10[0] = 0;
    }
    sub_23B2900((__int64)a1, (__int64)v4, v5, (__int64)byte_3F871B3, 0);
    if ( v8 != (_QWORD *)v10 )
      j_j___libc_free_0((unsigned __int64)v8);
  }
}
