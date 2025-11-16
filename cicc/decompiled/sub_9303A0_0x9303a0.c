// Function: sub_9303A0
// Address: 0x9303a0
//
__int64 __fastcall sub_9303A0(_QWORD *a1, __int64 a2)
{
  __int64 v3; // r15
  __int64 v4; // r14
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rcx
  char v10; // al
  __int64 v11; // rdi
  char v12; // bl
  unsigned int v13; // r12d
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // r15

  v3 = *(_QWORD *)(a2 + 72);
  v4 = *(_QWORD *)(v3 + 8);
  result = *(unsigned __int8 *)(v4 + 177);
  if ( (_BYTE)result == 4 )
    sub_91B8A0("block scope static variable initialization is not supported!", (_DWORD *)a2, 1);
  if ( (_BYTE)result && (_BYTE)result != 3 )
  {
    if ( (_BYTE)result != 2 )
      sub_91B8A0("unsupported dynamic initialization variant!", (_DWORD *)a2, 1);
    if ( *(_BYTE *)(v3 + 48) )
    {
      if ( sub_91B770(*(_QWORD *)(v4 + 120)) )
      {
        return sub_92FF60(a1, v3, v6, v7);
      }
      else
      {
        sub_91A3A0(a1[4] + 8LL, *(_QWORD *)(v4 + 120), v6, v7);
        v10 = *(_BYTE *)(v3 + 48);
        switch ( v10 )
        {
          case 2:
            v15 = *(_QWORD *)(v3 + 56);
            v17 = sub_91FFE0((__int64)a1, (const __m128i *)v15, 0, v9);
            break;
          case 3:
            v15 = *(_QWORD *)(v3 + 56);
            v17 = (__int64)sub_92F410((__int64)a1, v15);
            break;
          case 1:
            v15 = *(_QWORD *)(v4 + 120);
            v17 = sub_91DAD0(a1[4], v15, v8, v9);
            break;
          default:
            sub_91B8A0("unsupported dynamic initialization variant!", (_DWORD *)a2, 1);
        }
        v11 = *(_QWORD *)(v4 + 120);
        v12 = 0;
        if ( (*(_BYTE *)(v11 + 140) & 0xFB) == 8 )
        {
          v15 = dword_4F077C4 != 2;
          v12 = (sub_8D4C10(v11, v15) & 2) != 0;
        }
        v13 = sub_91CB50(v4, v15, v16);
        v14 = sub_9439D0(a1, v4);
        return sub_923130((__int64)a1, v17, v14, v13, v12);
      }
    }
  }
  return result;
}
