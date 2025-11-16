// Function: sub_63B800
// Address: 0x63b800
//
__int64 __fastcall sub_63B800(__int64 a1, __int64 a2)
{
  __int64 *v3; // rsi
  __int64 v4; // r15
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 *v9; // r12
  __int64 v10; // rdx
  __int64 v11; // rax
  int v13; // eax
  int v14; // eax
  int v15; // eax
  __int64 v16; // [rsp-8h] [rbp-58h]
  _DWORD v18[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v3 = (__int64 *)v18;
  v4 = qword_4F06BC0;
  sub_7296F0((unsigned int)(dword_4F04C64 - 1), v18);
  v5 = 776LL * dword_4F04C64;
  v6 = *(int *)(qword_4F04C68[0] + v5 - 376);
  if ( (_DWORD)v6 != -1 )
  {
    dword_4F04C58 = *(_DWORD *)(qword_4F04C68[0] + v5 - 376);
    unk_4F04C50 = *(_QWORD *)(qword_4F04C68[0] + 776 * v6 + 184);
  }
  qword_4F06BC0 = *(_QWORD *)(qword_4F04C68[0] + v5 + 496);
  v7 = sub_6E1C80(a2 + 328);
  *(_BYTE *)(a2 + 176) |= 8u;
  v8 = v7;
  v9 = (__int64 *)v7;
  if ( !(unsigned int)sub_6E1A80(v7) )
  {
    v8 = *(_QWORD *)(a2 + 288);
    if ( (*(_BYTE *)(a2 + 127) & 8) == 0 )
    {
      v3 = *(__int64 **)(a2 + 288);
      v8 = (__int64)v9;
      sub_694AA0(v9, v3, 0, dword_4D048B8, a2 + 136);
      v11 = *(_QWORD *)(a2 + 144);
      if ( v11 )
        goto LABEL_10;
LABEL_13:
      v11 = sub_72C9D0(v8, v3, v10);
      *(_QWORD *)(a2 + 144) = v11;
      goto LABEL_10;
    }
    if ( (*(_BYTE *)(a2 + 129) & 2) != 0 )
    {
      *(_BYTE *)(a2 + 176) |= 1u;
      v13 = sub_8D3880(v8);
      v10 = a2 + 136;
      if ( !v13
        || (v14 = sub_8D3880(*(_QWORD *)(a2 + 288)), v10 = a2 + 136, !v14)
        || (v3 = (__int64 *)(a2 + 288),
            v8 = (__int64)v9,
            v15 = sub_6320D0((__int64)v9, (__int64 *)(a2 + 288), a2 + 136, a2 + 136),
            v10 = a2 + 136,
            !v15) )
      {
        v3 = *(__int64 **)(a2 + 288);
        v8 = (__int64)v9;
        sub_694AA0(v9, v3, 0, dword_4D048B8, v10);
      }
    }
    else
    {
      if ( dword_4D04964 )
        *(_BYTE *)(a2 + 176) |= 0x80u;
      else
        *(_BYTE *)(a2 + 177) |= 1u;
      v3 = v9;
      sub_637180(v8, v9, (__m128i *)(a2 + 136), (_QWORD *)a2, dword_4D048B8, 0, (_QWORD *)(a2 + 24));
      v10 = v16;
    }
  }
  v11 = *(_QWORD *)(a2 + 144);
  if ( !v11 )
    goto LABEL_13;
LABEL_10:
  *(_QWORD *)(a1 + 8) = v11;
  sub_6E1990(v9);
  unk_4F04C50 = 0;
  dword_4F04C58 = -1;
  qword_4F06BC0 = v4;
  return sub_729730(v18[0]);
}
