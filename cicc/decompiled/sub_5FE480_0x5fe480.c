// Function: sub_5FE480
// Address: 0x5fe480
//
__int64 __fastcall sub_5FE480(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r15
  char v10; // al
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  _QWORD *v14; // rdx
  _QWORD *v15; // rax
  __int64 result; // rax
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // [rsp+0h] [rbp-80h]
  __m128i v21[7]; // [rsp+10h] [rbp-70h] BYREF

  v6 = *a1;
  v7 = sub_7259C0(7);
  v8 = *(_QWORD *)(v7 + 168);
  v9 = v7;
  v10 = *(_BYTE *)(a2 + 560);
  v20 = v8;
  if ( (v10 & 2) != 0 )
  {
    v19 = sub_72CBE0();
    v12 = v20;
    *(_QWORD *)(v9 + 160) = v19;
    *(_BYTE *)(v20 + 17) |= 1u;
  }
  else
  {
    if ( (v10 & 8) != 0 )
    {
      v18 = sub_72CBE0();
      v12 = v20;
      *(_QWORD *)(v9 + 160) = v18;
      *(_QWORD *)v20 = a4;
      *(_BYTE *)(v20 + 17) |= 2u;
      if ( a4 )
        goto LABEL_5;
      goto LABEL_22;
    }
    v11 = sub_72D600(v6);
    v12 = v20;
    *(_QWORD *)(v9 + 160) = v11;
  }
  *(_QWORD *)v12 = a4;
  if ( a4 )
  {
LABEL_5:
    *(_DWORD *)(a4 + 36) = 1;
    *(_BYTE *)(v12 + 21) |= 1u;
    *(_BYTE *)(v12 + 16) |= 2u;
    *(_QWORD *)(v12 + 40) = v6;
    sub_8DCB20(v9);
    goto LABEL_6;
  }
LABEL_22:
  *(_BYTE *)(v12 + 21) |= 1u;
  *(_BYTE *)(v12 + 16) |= 2u;
  *(_QWORD *)(v12 + 40) = v6;
LABEL_6:
  sub_7325D0(v9, &unk_4F077C8);
  *(_QWORD *)(a2 + 288) = v9;
  *(_BYTE *)(a3 + 64) |= 2u;
  if ( dword_4D048B8 )
    *(_QWORD *)(a3 + 24) = *(_QWORD *)(v6 + 64);
  if ( (*(_BYTE *)(a2 + 560) & 0xA) != 0 )
  {
    sub_878710(*(_QWORD *)v6, v21);
    if ( (*(_BYTE *)(a2 + 560) & 2) != 0 )
      sub_87A680(v21, v6 + 64, 0);
    else
      sub_87A530(v21, 0);
  }
  else
  {
    sub_87A720(15, v21, v6 + 64);
  }
  sub_5FBCD0(v21, a3, (__int64)a1, (__int64 *)a2, 1u);
  if ( dword_4F04C64 == -1
    || (v13 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v13 + 7) & 1) == 0)
    || dword_4F04C44 == -1 && (*(_BYTE *)(v13 + 6) & 2) == 0 )
  {
    if ( (*(_BYTE *)(a3 + 65) & 8) == 0 )
      sub_87E280(a3 + 8);
    v13 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  }
  v14 = *(_QWORD **)(v13 + 24);
  v15 = (_QWORD *)(v13 + 32);
  if ( !v14 )
    v14 = v15;
  for ( ; *(_BYTE *)(v6 + 140) == 12; v6 = *(_QWORD *)(v6 + 160) )
    ;
  **(_QWORD **)(*(_QWORD *)v6 + 96LL) = *v14;
  result = dword_4F068EC;
  if ( dword_4F068EC )
  {
    result = *(_QWORD *)a2;
    v17 = *(_QWORD *)(*(_QWORD *)a2 + 88LL);
    if ( (*(_BYTE *)(v17 + 195) & 8) == 0 )
      return sub_89A080(v17);
  }
  return result;
}
