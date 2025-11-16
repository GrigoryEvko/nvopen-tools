// Function: sub_865B60
// Address: 0x865b60
//
__int64 __fastcall sub_865B60(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  __int64 v4; // r14
  unsigned int v5; // ebx
  __int64 v6; // r15
  __int64 v7; // r13
  __int64 v8; // rdi
  __int64 v9; // r10
  int v10; // ebx
  __int64 v11; // rax
  __int64 v12; // [rsp-10h] [rbp-50h]
  __int64 v13; // [rsp+0h] [rbp-40h]
  __int64 v14; // [rsp+0h] [rbp-40h]
  int v15; // [rsp+Ch] [rbp-34h]

  if ( (*(_BYTE *)(a1 + 178) & 4) != 0 )
    return sub_866000(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL), 1, 0);
  v4 = *(_QWORD *)a1;
  v5 = a2;
  v6 = sub_878920(*(_QWORD *)a1);
  v7 = sub_892330(a1);
  switch ( *(_BYTE *)(v6 + 80) )
  {
    case 4:
    case 5:
      v8 = *(_QWORD *)(*(_QWORD *)(v6 + 96) + 80LL);
      goto LABEL_5;
    case 6:
      v8 = *(_QWORD *)(*(_QWORD *)(v6 + 96) + 32LL);
      goto LABEL_5;
    case 9:
    case 0xA:
      v9 = *(_QWORD *)(sub_892400(*(_QWORD *)(*(_QWORD *)(v6 + 96) + 56LL)) + 32);
      if ( !a2 )
        goto LABEL_16;
      goto LABEL_6;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v8 = *(_QWORD *)(v6 + 88);
      goto LABEL_5;
    default:
      v8 = 0;
LABEL_5:
      v9 = *(_QWORD *)(sub_892400(v8) + 32);
      if ( a2 )
      {
LABEL_6:
        v15 = dword_4F04C64;
        v10 = unk_4F04C2C;
        v11 = *(_QWORD *)(a1 + 40);
        if ( (*(_BYTE *)(a1 + 89) & 4) != 0 )
        {
          v13 = v9;
          sub_8646E0(*(_QWORD *)(v11 + 32), 0);
          v9 = v13;
        }
        else if ( v11 && *(_BYTE *)(v11 + 28) == 3 )
        {
          v14 = v9;
          sub_864360(*(_QWORD *)(v11 + 32), 1);
          v9 = v14;
        }
        if ( v9 )
          sub_85E1C0(v9, a1, 0, v4, v6, v7, 1u);
        if ( dword_4F04C64 == -1 )
        {
          MEMORY[7] &= ~0x20u;
          BUG();
        }
        result = qword_4F04C68[0] + 776LL * dword_4F04C64;
        *(_BYTE *)(result + 7) |= 0x20u;
        *(_DWORD *)(result + 576) = v10;
        *(_DWORD *)(result + 572) = v15;
      }
      else
      {
LABEL_16:
        if ( (unsigned __int8)(*(_BYTE *)(v4 + 80) - 4) <= 1u && *(char *)(*(_QWORD *)(v4 + 88) + 177LL) < 0 )
          v5 = 2;
        sub_864700(v9, a1, 0, v4, v6, v7, 0, v5);
        result = v12;
      }
      break;
  }
  return result;
}
