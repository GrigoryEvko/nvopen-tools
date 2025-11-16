// Function: sub_6D6DB0
// Address: 0x6d6db0
//
__int64 __fastcall sub_6D6DB0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r15
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // r11
  __int64 v8; // rdx
  int v9; // eax
  __int64 result; // rax
  __int64 v11; // rax
  _DWORD *v12; // rax
  __int64 v13; // [rsp+8h] [rbp-108h]
  int v14; // [rsp+10h] [rbp-100h]
  _DWORD *v15; // [rsp+18h] [rbp-F8h]
  __int64 v16; // [rsp+28h] [rbp-E8h] BYREF
  FILE v17; // [rsp+30h] [rbp-E0h] BYREF

  v2 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v3 = *(_QWORD *)(v2 + 624);
  *(_QWORD *)(v2 + 624) = a1;
  if ( !dword_4F077BC )
    goto LABEL_9;
  v4 = *(_QWORD *)(a1 + 328);
  if ( v4 )
  {
    if ( *(_BYTE *)(v4 + 8) != 1 )
      goto LABEL_4;
LABEL_9:
    *(_BYTE *)(a1 + 176) |= 4u;
    if ( !word_4D04898 )
      *(_BYTE *)(a1 + 178) |= 0x40u;
    goto LABEL_11;
  }
  if ( word_4F06418[0] == 73 )
    goto LABEL_9;
LABEL_4:
  if ( !word_4D04898
    || (v5 = *(_QWORD *)a1) == 0
    || *(_BYTE *)(v5 + 80) != 9
    || (*(_BYTE *)(*(_QWORD *)(v5 + 88) + 170LL) & 0x20) == 0 )
  {
    sub_6D6AC0(*(_QWORD *)(a1 + 288), a1, a2);
    goto LABEL_19;
  }
  *(_BYTE *)(a1 + 176) |= 4u;
LABEL_11:
  sub_6E2250(&v17._IO_read_end, &v16, 1, 1, a1, 0);
  v6 = sub_6BB5A0(1, 0);
  sub_694AA0(v6, *(_QWORD *)(a1 + 288), 1, 1, a1 + 136);
  if ( (*(_BYTE *)(a1 + 177) & 2) != 0 )
  {
    sub_72C970(a2);
  }
  else
  {
    v7 = *(_QWORD *)(a1 + 144);
    if ( v7 )
    {
      if ( (dword_4F04C44 != -1
         || (v11 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v11 + 6) & 6) != 0)
         || *(_BYTE *)(v11 + 4) == 12)
        && (unsigned __int8)(*(_BYTE *)(v7 + 48) - 3) <= 1u )
      {
        sub_70FD90(*(_QWORD *)(v7 + 56), a2);
      }
      else
      {
        v8 = *(_QWORD *)(a1 + 288);
        v13 = *(_QWORD *)(a1 + 144);
        *(_QWORD *)&v17._flags = 0;
        v14 = v8;
        v17._IO_read_ptr = 0;
        v9 = sub_6E1A20(v6);
        if ( !(unsigned int)sub_7A1C60(v13, v9, v14, 1, a2, (unsigned int)&v17, 0) )
        {
          v12 = (_DWORD *)sub_6E1A20(v6);
          v15 = sub_67D9D0(0x1Cu, v12);
          sub_67E370((__int64)v15, (const __m128i *)&v17);
          sub_685910((__int64)v15, &v17);
          sub_72C970(a2);
        }
        sub_67E3D0(&v17);
      }
    }
    else
    {
      sub_72A510(*(_QWORD *)(a1 + 136), a2);
    }
  }
  sub_6E1990(v6);
  sub_6E2C70(v16, 1, a1, 0);
LABEL_19:
  result = qword_4F04C68[0] + 776LL * dword_4F04C64;
  *(_QWORD *)(result + 624) = v3;
  return result;
}
