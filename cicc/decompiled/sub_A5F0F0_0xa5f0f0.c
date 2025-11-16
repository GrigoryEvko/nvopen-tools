// Function: sub_A5F0F0
// Address: 0xa5f0f0
//
__int64 __fastcall sub_A5F0F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  unsigned __int8 v5; // al
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // r8
  unsigned __int8 v9; // al
  __int64 v10; // rdx
  __int64 v11; // rcx
  unsigned __int8 v12; // al
  __int64 *v13; // rdx
  unsigned int v14; // eax
  unsigned __int8 v15; // al
  __int64 v16; // rdx
  unsigned __int8 v17; // al
  __int64 v18; // rdx
  unsigned __int8 v19; // al
  __int64 v20; // rdx
  unsigned __int8 v21; // al
  __int64 v22; // rdx
  unsigned __int8 v23; // al
  __int64 v24; // r13
  __int64 v26; // [rsp+0h] [rbp-40h] BYREF
  char v27; // [rsp+8h] [rbp-38h]
  char *v28; // [rsp+10h] [rbp-30h]
  __int64 v29; // [rsp+18h] [rbp-28h]

  v4 = a2 - 16;
  sub_904010(a1, "!DISubrangeType(");
  v26 = a1;
  v28 = ", ";
  v5 = *(_BYTE *)(a2 - 16);
  v27 = 1;
  v29 = a3;
  if ( (v5 & 2) != 0 )
  {
    v6 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 16LL);
    if ( v6 )
    {
LABEL_3:
      v6 = sub_B91420(v6, "!DISubrangeType(");
      v8 = v7;
      goto LABEL_4;
    }
  }
  else
  {
    v6 = *(_QWORD *)(a2 - 8LL * ((v5 >> 2) & 0xF));
    if ( v6 )
      goto LABEL_3;
  }
  v8 = 0;
LABEL_4:
  sub_A53660(&v26, "name", 4u, v6, v8, 1);
  v9 = *(_BYTE *)(a2 - 16);
  if ( (v9 & 2) != 0 )
    v10 = *(_QWORD *)(a2 - 32);
  else
    v10 = v4 - 8LL * ((v9 >> 2) & 0xF);
  sub_A5CC00((__int64)&v26, "scope", 5u, *(_QWORD *)(v10 + 8), 1);
  v11 = a2;
  if ( *(_BYTE *)a2 != 16 )
  {
    v12 = *(_BYTE *)(a2 - 16);
    if ( (v12 & 2) != 0 )
      v13 = *(__int64 **)(a2 - 32);
    else
      v13 = (__int64 *)(v4 - 8LL * ((v12 >> 2) & 0xF));
    v11 = *v13;
  }
  sub_A5CC00((__int64)&v26, "file", 4u, v11, 1);
  sub_A537C0((__int64)&v26, "line", 4u, *(_DWORD *)(a2 + 16), 1);
  sub_A539C0((__int64)&v26, "size", 4u, *(_QWORD *)(a2 + 24));
  v14 = sub_AF18D0(a2);
  sub_A537C0((__int64)&v26, "align", 5u, v14, 1);
  sub_A53C60(&v26, "flags", 5u, *(_DWORD *)(a2 + 20));
  v15 = *(_BYTE *)(a2 - 16);
  if ( (v15 & 2) != 0 )
    v16 = *(_QWORD *)(a2 - 32);
  else
    v16 = v4 - 8LL * ((v15 >> 2) & 0xF);
  sub_A5CC00((__int64)&v26, "baseType", 8u, *(_QWORD *)(v16 + 24), 0);
  v17 = *(_BYTE *)(a2 - 16);
  if ( (v17 & 2) != 0 )
    v18 = *(_QWORD *)(a2 - 32);
  else
    v18 = v4 - 8LL * ((v17 >> 2) & 0xF);
  sub_A5CC00((__int64)&v26, "lowerBound", 0xAu, *(_QWORD *)(v18 + 32), 1);
  v19 = *(_BYTE *)(a2 - 16);
  if ( (v19 & 2) != 0 )
    v20 = *(_QWORD *)(a2 - 32);
  else
    v20 = v4 - 8LL * ((v19 >> 2) & 0xF);
  sub_A5CC00((__int64)&v26, "upperBound", 0xAu, *(_QWORD *)(v20 + 40), 1);
  v21 = *(_BYTE *)(a2 - 16);
  if ( (v21 & 2) != 0 )
    v22 = *(_QWORD *)(a2 - 32);
  else
    v22 = v4 - 8LL * ((v21 >> 2) & 0xF);
  sub_A5CC00((__int64)&v26, "stride", 6u, *(_QWORD *)(v22 + 48), 1);
  v23 = *(_BYTE *)(a2 - 16);
  if ( (v23 & 2) != 0 )
    v24 = *(_QWORD *)(a2 - 32);
  else
    v24 = v4 - 8LL * ((v23 >> 2) & 0xF);
  sub_A5CC00((__int64)&v26, "bias", 4u, *(_QWORD *)(v24 + 56), 1);
  return sub_904010(a1, ")");
}
