// Function: sub_1290F50
// Address: 0x1290f50
//
__int64 __fastcall sub_1290F50(__int64 *a1, __int64 a2)
{
  __int64 v3; // r15
  __int64 v4; // r14
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // rbx
  char v8; // al
  __int64 v9; // rdi
  char v10; // bl
  unsigned int v11; // r12d
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // r15
  char *v16; // rax

  v3 = *(_QWORD *)(a2 + 72);
  v4 = *(_QWORD *)(v3 + 8);
  result = *(unsigned __int8 *)(v4 + 177);
  if ( (_BYTE)result == 4 )
    sub_127B550("block scope static variable initialization is not supported!", (_DWORD *)a2, 1);
  if ( (_BYTE)result && (_BYTE)result != 3 )
  {
    if ( (_BYTE)result != 2 )
      sub_127B550("unsupported dynamic initialization variant!", (_DWORD *)a2, 1);
    if ( *(_BYTE *)(v3 + 48) )
    {
      if ( sub_127B420(*(_QWORD *)(v4 + 120)) )
      {
        return sub_1290BA0(a1, v3, v6);
      }
      else
      {
        v7 = sub_127A040(a1[4] + 8, *(_QWORD *)(v4 + 120));
        v8 = *(_BYTE *)(v3 + 48);
        switch ( v8 )
        {
          case 2:
            v13 = *(_QWORD *)(v3 + 56);
            v15 = sub_127F650((__int64)a1, (const __m128i *)v13, 0);
            break;
          case 3:
            v16 = sub_128F980((__int64)a1, *(_QWORD *)(v3 + 56));
            v13 = v7;
            v15 = sub_1289860(a1, v7, v16);
            break;
          case 1:
            v13 = *(_QWORD *)(v4 + 120);
            v15 = sub_127D2A0(a1[4], v13);
            break;
          default:
            sub_127B550("unsupported dynamic initialization variant!", (_DWORD *)a2, 1);
        }
        v9 = *(_QWORD *)(v4 + 120);
        v10 = 0;
        if ( (*(_BYTE *)(v9 + 140) & 0xFB) == 8 )
        {
          v13 = dword_4F077C4 != 2;
          v10 = (sub_8D4C10(v9, v13) & 2) != 0;
        }
        v11 = sub_127C800(v4, v13, v14);
        v12 = sub_12A2A10(a1, v4);
        return sub_1280F50(a1, v15, v12, v11, v10);
      }
    }
  }
  return result;
}
