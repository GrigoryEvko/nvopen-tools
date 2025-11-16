// Function: sub_5FE150
// Address: 0x5fe150
//
__int64 __fastcall sub_5FE150(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r13
  __int64 v8; // rdx
  __int64 v9; // rax
  _QWORD *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // r14
  _BYTE *v14; // rax
  __int64 result; // rax
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // rdx
  __int64 i; // rdi
  __int64 v20; // r8
  __int64 v21; // rax
  char v22; // r10
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rsi
  __int64 v28; // [rsp+8h] [rbp-338h]
  char v29; // [rsp+8h] [rbp-338h]
  __m128i v30[4]; // [rsp+10h] [rbp-330h] BYREF
  char v31[8]; // [rsp+50h] [rbp-2F0h] BYREF
  char v32[16]; // [rsp+58h] [rbp-2E8h] BYREF
  __int64 v33; // [rsp+68h] [rbp-2D8h]
  char v34; // [rsp+90h] [rbp-2B0h]
  char v35; // [rsp+91h] [rbp-2AFh]
  __int64 j; // [rsp+A0h] [rbp-2A0h]
  __int64 v37[80]; // [rsp+C0h] [rbp-280h] BYREF

  v7 = *a1;
  v8 = *(_QWORD *)(*a1 + 168);
  v9 = *(_QWORD *)(v8 + 152);
  v10 = *(_QWORD **)(v8 + 136);
  v11 = *(_QWORD *)(v9 + 144);
  if ( !v11 )
  {
    if ( v10 )
    {
      v12 = 0;
      v13 = 0;
      goto LABEL_14;
    }
    v12 = 0;
    sub_69C3A0(0, *a1, 0, a4, a5, a6);
    goto LABEL_24;
  }
  v12 = 0;
  v13 = 0;
  do
  {
    if ( *(_BYTE *)(v11 + 174) == 5 )
    {
      a4 = *(unsigned __int8 *)(v11 + 176);
      if ( (_BYTE)a4 == 30 )
      {
        v13 = v11;
      }
      else if ( (_BYTE)a4 == 34 && (*(_BYTE *)(v11 + 206) & 8) != 0 )
      {
        v12 = v11;
      }
    }
    v11 = *(_QWORD *)(v11 + 112);
  }
  while ( v11 );
  while ( v10 )
  {
LABEL_14:
    v14 = (_BYTE *)v10[1];
    if ( v14[174] == 5 )
    {
      a4 = (unsigned __int8)v14[176];
      if ( (_BYTE)a4 == 30 )
      {
        v13 = v10[1];
      }
      else if ( (_BYTE)a4 == 34 && (v14[206] & 8) != 0 )
      {
        v12 = v10[1];
      }
    }
    v10 = (_QWORD *)*v10;
  }
  result = sub_69C3A0(v12, *a1, 0, a4, a5, a6);
  if ( !v13 )
  {
LABEL_24:
    sub_5E4C60((__int64)v37, (_QWORD *)(v7 + 64));
    sub_87E3B0(v31);
    v34 |= 2u;
    v18 = v7 + 64;
    if ( dword_4D048B8 )
    {
      v33 = *(_QWORD *)(v7 + 64);
      sub_894C00(*(_QWORD *)v12, v7 + 64, v18, v16, v17);
      v18 = v7 + 64;
    }
    sub_87A720(30, v30, v18);
    for ( i = *(_QWORD *)(v12 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v28 = sub_73EDA0(i, 0);
    *(_QWORD *)(v28 + 160) = sub_72C390();
    v37[36] = v28;
    v37[35] = v28;
    v21 = *(_QWORD *)(v12 + 152);
    for ( j = v28; *(_BYTE *)(v21 + 140) == 12; v21 = *(_QWORD *)(v21 + 160) )
      ;
    if ( *(_QWORD *)(*(_QWORD *)(v21 + 168) + 40LL) )
    {
      v22 = *((_BYTE *)a1 + 12);
      *((_BYTE *)a1 + 12) = *(_BYTE *)(v12 + 88) & 3;
      v29 = v22;
      sub_5FBCD0(v30, (__int64)v31, (__int64)a1, v37, 1u);
      *((_BYTE *)a1 + 12) = v29;
    }
    else
    {
      sub_5F1000(v30, *a1, (__int64)v31, v37, v20);
    }
    v27 = *(_QWORD *)(v37[0] + 88);
    *(_BYTE *)(v27 + 193) |= 0x10u;
    sub_691E30(v7, v27, v23, v24, v25, v26);
    result = dword_4F04C64;
    if ( dword_4F04C64 == -1
      || (result = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(result + 7) & 1) == 0)
      || dword_4F04C44 == -1 && (*(_BYTE *)(result + 6) & 2) == 0 )
    {
      if ( (v35 & 8) == 0 )
        return sub_87E280(v32);
    }
  }
  return result;
}
