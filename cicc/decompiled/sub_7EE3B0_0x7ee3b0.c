// Function: sub_7EE3B0
// Address: 0x7ee3b0
//
void __fastcall sub_7EE3B0(__int64 a1)
{
  __m128i *v1; // r13
  __int64 v2; // rax
  __int64 v3; // rbx
  unsigned int v4; // r14d
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  _QWORD *v9; // rax
  __m128i *v10; // r14
  __int64 v11; // r12
  _BYTE *v12; // r12
  __int64 v13; // rax
  _QWORD *v14; // [rsp+8h] [rbp-A8h] BYREF
  _BYTE v15[32]; // [rsp+10h] [rbp-A0h] BYREF
  _BYTE v16[128]; // [rsp+30h] [rbp-80h] BYREF

  v1 = *(__m128i **)(a1 + 56);
  v2 = sub_7E2C20((__int64)v1);
  v3 = v2;
  if ( v2 )
  {
    if ( *(_BYTE *)(v2 + 40) == 25 )
    {
      v9 = *(_QWORD **)(v2 + 48);
      if ( v9 && (unsigned int)sub_8D2600(*v9) )
      {
        *(_BYTE *)(v3 + 40) = 0;
        v3 = 0;
      }
    }
    else
    {
      v3 = 0;
    }
  }
  v4 = dword_4D04380;
  dword_4D04380 = 0;
  sub_7E1A40((__int64)v16, 0, 0, (__int64)&v14);
  if ( dword_4F077C4 == 2 )
    sub_7EDF20(v1, 0, 1, 0, 0);
  else
    sub_7DA050(v1, 0, v5, v6, v7, v8);
  sub_7E1B40(v14);
  dword_4D04380 = v4;
  if ( v3 )
  {
    if ( *(_BYTE *)(v3 + 40) == 11 && *(_BYTE *)(sub_7E2C20(v3) + 40) == 25 )
      sub_7E2C40((const __m128i *)v3);
    if ( *(_QWORD *)(v3 + 16) )
    {
      if ( *(_BYTE *)(v3 + 40) == 25 )
      {
        v10 = sub_7E7C20(**(_QWORD **)(v3 + 48), *(_QWORD *)(v1[5].m128i_i64[0] + 8), 0, 0);
        v11 = sub_7E2BE0((__int64)v10, *(_QWORD *)(v3 + 48));
        *(_BYTE *)(*(_QWORD *)(v3 + 48) + 25LL) &= ~4u;
        sub_7304E0(v11);
        sub_7268E0(v3, 0);
        *(_QWORD *)(v3 + 48) = v11;
        v12 = sub_726B30(25);
        *((_QWORD *)v12 + 6) = sub_73E830((__int64)v10);
        v13 = sub_7E2C20((__int64)v1);
        sub_7E1720(v13, (__int64)v15);
        sub_7E6810((__int64)v12, (__int64)v15, 1);
      }
    }
  }
}
