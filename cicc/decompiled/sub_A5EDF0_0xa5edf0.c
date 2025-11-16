// Function: sub_A5EDF0
// Address: 0xa5edf0
//
__int64 __fastcall sub_A5EDF0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  unsigned __int8 v5; // al
  __int64 *v6; // rdx
  __int64 v7; // r15
  unsigned __int8 v8; // al
  __int64 v9; // rdx
  __int64 v10; // r15
  unsigned __int8 v11; // al
  __int64 v12; // rdx
  __int64 v13; // r15
  unsigned __int8 v14; // al
  __int64 v15; // r14
  __int64 v16; // r14
  __int64 v18; // [rsp+40h] [rbp-50h] BYREF
  char v19; // [rsp+48h] [rbp-48h]
  char *v20; // [rsp+50h] [rbp-40h]
  __int64 v21; // [rsp+58h] [rbp-38h]

  v4 = a2 - 16;
  sub_904010(a1, "!DIGenericSubrange(");
  v18 = a1;
  v20 = ", ";
  v5 = *(_BYTE *)(a2 - 16);
  v19 = 1;
  v21 = a3;
  if ( (v5 & 2) != 0 )
    v6 = *(__int64 **)(a2 - 32);
  else
    v6 = (__int64 *)(v4 - 8LL * ((v5 >> 2) & 0xF));
  v7 = *v6;
  if ( *v6
    && *(_BYTE *)v7 == 7
    && (unsigned __int8)((unsigned __int64)sub_AF4F20(*v6) >> 32)
    && !(unsigned int)sub_AF4F20(v7) )
  {
    sub_A538D0((__int64)&v18, "count", 5u, *(_QWORD *)(*(_QWORD *)(v7 + 16) + 8LL));
  }
  else
  {
    sub_A5CC00((__int64)&v18, "count", 5u, v7, 1);
  }
  v8 = *(_BYTE *)(a2 - 16);
  if ( (v8 & 2) != 0 )
    v9 = *(_QWORD *)(a2 - 32);
  else
    v9 = v4 - 8LL * ((v8 >> 2) & 0xF);
  v10 = *(_QWORD *)(v9 + 8);
  if ( v10
    && *(_BYTE *)v10 == 7
    && (unsigned __int8)((unsigned __int64)sub_AF4F20(*(_QWORD *)(v9 + 8)) >> 32)
    && !(unsigned int)sub_AF4F20(v10) )
  {
    sub_A538D0((__int64)&v18, "lowerBound", 0xAu, *(_QWORD *)(*(_QWORD *)(v10 + 16) + 8LL));
  }
  else
  {
    sub_A5CC00((__int64)&v18, "lowerBound", 0xAu, v10, 1);
  }
  v11 = *(_BYTE *)(a2 - 16);
  if ( (v11 & 2) != 0 )
    v12 = *(_QWORD *)(a2 - 32);
  else
    v12 = v4 - 8LL * ((v11 >> 2) & 0xF);
  v13 = *(_QWORD *)(v12 + 16);
  if ( v13
    && *(_BYTE *)v13 == 7
    && (unsigned __int8)((unsigned __int64)sub_AF4F20(*(_QWORD *)(v12 + 16)) >> 32)
    && !(unsigned int)sub_AF4F20(v13) )
  {
    sub_A538D0((__int64)&v18, "upperBound", 0xAu, *(_QWORD *)(*(_QWORD *)(v13 + 16) + 8LL));
  }
  else
  {
    sub_A5CC00((__int64)&v18, "upperBound", 0xAu, v13, 1);
  }
  v14 = *(_BYTE *)(a2 - 16);
  if ( (v14 & 2) != 0 )
    v15 = *(_QWORD *)(a2 - 32);
  else
    v15 = v4 - 8LL * ((v14 >> 2) & 0xF);
  v16 = *(_QWORD *)(v15 + 24);
  if ( v16
    && *(_BYTE *)v16 == 7
    && (unsigned __int8)((unsigned __int64)sub_AF4F20(v16) >> 32)
    && !(unsigned int)sub_AF4F20(v16) )
  {
    sub_A538D0((__int64)&v18, "stride", 6u, *(_QWORD *)(*(_QWORD *)(v16 + 16) + 8LL));
  }
  else
  {
    sub_A5CC00((__int64)&v18, "stride", 6u, v16, 1);
  }
  return sub_904010(a1, ")");
}
