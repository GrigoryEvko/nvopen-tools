// Function: sub_7FA230
// Address: 0x7fa230
//
void __fastcall sub_7FA230(__int64 a1, int *a2)
{
  __int64 v2; // r12
  _QWORD *v3; // r14
  _BYTE *v4; // rax
  __int64 v5; // r8
  __int64 v6; // r9
  _QWORD *v7; // rax
  _QWORD *v8; // rbx
  _BYTE *v9; // rax
  const __m128i *v10[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *(_QWORD *)(a1 + 88);
  if ( *(_BYTE *)(v2 + 136) != 2 )
  {
    v10[0] = (const __m128i *)sub_724DC0();
    sub_72BAF0((__int64)v10[0]->m128i_i64, 0, 5u);
    if ( (unsigned int)(*a2 - 3) > 2 )
    {
      v7 = sub_725A70(2u);
      *((_BYTE *)v7 + 49) |= 2u;
      v8 = v7;
      v7[1] = v2;
      *(_BYTE *)(v2 + 177) = 2;
      *(_QWORD *)(v2 + 184) = v7;
      v7[7] = sub_724E50((__int64 *)v10, 0);
      v9 = sub_726B30(17);
      *((_QWORD *)v9 + 9) = v8;
      sub_7E6810((__int64)v9, (__int64)a2, 1);
    }
    else
    {
      v3 = sub_73A720(v10[0], 0);
      v4 = sub_731250(v2);
      sub_7E6A80(v4, 0x49u, (__int64)v3, a2, v5, v6);
    }
    if ( v10[0] )
      sub_724E30((__int64)v10);
  }
}
