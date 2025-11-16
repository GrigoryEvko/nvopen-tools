// Function: sub_2B08AC0
// Address: 0x2b08ac0
//
__int64 __fastcall sub_2B08AC0(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rax
  __int64 v5; // rdi
  unsigned __int64 v6; // rax
  __int64 v7; // rdi
  unsigned __int64 v8; // rax
  __m128i v9; // [rsp-B8h] [rbp-B8h] BYREF
  __int64 v10; // [rsp-A8h] [rbp-A8h]
  __int64 v11; // [rsp-A0h] [rbp-A0h]
  __int64 v12; // [rsp-98h] [rbp-98h]
  __int64 v13; // [rsp-90h] [rbp-90h]
  __int64 v14; // [rsp-88h] [rbp-88h]
  __int64 v15; // [rsp-80h] [rbp-80h]
  __int16 v16; // [rsp-78h] [rbp-78h]
  __m128i v17; // [rsp-68h] [rbp-68h] BYREF
  __int64 v18; // [rsp-58h] [rbp-58h]
  __int64 v19; // [rsp-50h] [rbp-50h]
  __int64 v20; // [rsp-48h] [rbp-48h]
  __int64 v21; // [rsp-40h] [rbp-40h]
  __int64 v22; // [rsp-38h] [rbp-38h]
  __int64 v23; // [rsp-30h] [rbp-30h]
  __int16 v24; // [rsp-28h] [rbp-28h]

  result = 0;
  if ( *(_BYTE *)a2 == 82 )
  {
    if ( sub_B532B0(*(_WORD *)(a2 + 2) & 0x3F) )
      return 1;
    v4 = *a1;
    v5 = *(_QWORD *)(a2 - 64);
    v16 = 257;
    v6 = *(_QWORD *)(v4 + 3344);
    v10 = 0;
    v9 = (__m128i)v6;
    v11 = 0;
    v12 = 0;
    v13 = 0;
    v14 = 0;
    v15 = 0;
    if ( (unsigned __int8)sub_9AC470(v5, &v9, 0) )
    {
      v7 = *(_QWORD *)(a2 - 32);
      v8 = *(_QWORD *)(*a1 + 3344);
      v18 = 0;
      v19 = 0;
      v17 = (__m128i)v8;
      v20 = 0;
      v21 = 0;
      v22 = 0;
      v23 = 0;
      v24 = 257;
      return (unsigned int)sub_9AC470(v7, &v17, 0) ^ 1;
    }
    else
    {
      return 1;
    }
  }
  return result;
}
