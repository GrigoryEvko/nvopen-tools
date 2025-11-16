// Function: sub_2143C70
// Address: 0x2143c70
//
unsigned __int64 __fastcall sub_2143C70(
        __int64 *a1,
        __int64 a2,
        _DWORD *a3,
        _DWORD *a4,
        __m128i a5,
        double a6,
        __m128i a7)
{
  unsigned __int64 *v9; // rax
  unsigned __int64 v10; // r15
  __int64 v11; // rcx
  __int64 v12; // rax
  char v13; // dl
  __int64 v14; // rax
  bool v15; // al
  __int64 v16; // rdx
  _QWORD *v17; // rax
  bool v18; // zf
  _DWORD *v19; // rax
  __int64 v21; // [rsp+8h] [rbp-48h]
  _BYTE v22[8]; // [rsp+10h] [rbp-40h] BYREF
  __int64 v23; // [rsp+18h] [rbp-38h]

  v9 = *(unsigned __int64 **)(a2 + 32);
  v10 = *v9;
  v11 = v9[1];
  v12 = *(_QWORD *)(*v9 + 40) + 16LL * *((unsigned int *)v9 + 2);
  v13 = *(_BYTE *)v12;
  v14 = *(_QWORD *)(v12 + 8);
  v22[0] = v13;
  v23 = v14;
  if ( v13 )
  {
    v15 = (unsigned __int8)(v13 - 14) <= 0x47u || (unsigned __int8)(v13 - 2) <= 5u;
  }
  else
  {
    v21 = v11;
    v15 = sub_1F58CF0((__int64)v22);
    v11 = v21;
  }
  if ( v15 )
    sub_20174B0((__int64)a1, v10, v11, a3, a4);
  else
    sub_2016B80((__int64)a1, v10, v11, a3, a4);
  v16 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL) + 88LL);
  v17 = *(_QWORD **)(v16 + 24);
  if ( *(_DWORD *)(v16 + 32) > 0x40u )
    v17 = (_QWORD *)*v17;
  v18 = v17 == 0;
  v19 = a3;
  if ( !v18 )
    v19 = a4;
  return sub_200D960(a1, *(_QWORD *)v19, *((_QWORD *)v19 + 1), (__int64)a3, (__int64)a4, a5, a6, a7);
}
