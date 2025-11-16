// Function: sub_324B9D0
// Address: 0x324b9d0
//
__int64 __fastcall sub_324B9D0(__int64 *a1, unsigned __int64 **a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rdi
  _QWORD *v9; // rax
  int v10; // ebx
  bool v11; // r15
  __int64 v12; // rsi
  __int64 v13; // rbx
  __int64 v14; // rdi
  __int64 result; // rax
  unsigned int v16; // [rsp+Ch] [rbp-34h]

  v8 = a1[26];
  v9 = *(_QWORD **)a3;
  v10 = *(_DWORD *)(v8 + 3760);
  if ( (*(_QWORD *)a3
     || (*(_BYTE *)(a3 + 9) & 0x70) == 0x20
     && *(char *)(a3 + 8) >= 0
     && (*(_BYTE *)(a3 + 8) |= 8u, v9 = sub_E807D0(*(_QWORD *)(a3 + 24)), *(_QWORD *)a3 = v9, v8 = a1[26], v9))
    && (unsigned int)(v10 - 3) <= 1
    && off_4C5D170 != (_UNKNOWN *)v9 )
  {
    v11 = 0;
    v12 = a3;
    v13 = sub_3222A80(v8, v9[1]);
    v14 = a1[26] + 4840;
    if ( v13 )
    {
      v12 = v13;
      v11 = a3 != v13;
    }
  }
  else
  {
    v14 = v8 + 4840;
    v12 = a3;
    v11 = 0;
    v13 = 0;
  }
  v16 = sub_37291A0(v14, v12, 0, a4, a5);
  if ( (unsigned __int16)sub_3220AA0(a1[26]) > 4u )
  {
    sub_3249B00(a1, a2, 11, 161);
    result = sub_3249B00(a1, a2, 27, v16);
    if ( !v11 )
      return result;
LABEL_14:
    sub_3249B00(a1, a2, 11, 12);
    sub_324B8B0(a1, a2, 0, a3, v13);
    return sub_3249B00(a1, a2, 11, 34);
  }
  sub_3249B00(a1, a2, 11, 251);
  result = sub_3249B00(a1, a2, 7937, v16);
  if ( v11 )
    goto LABEL_14;
  return result;
}
