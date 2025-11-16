// Function: sub_37ADB10
// Address: 0x37adb10
//
__m128i *__fastcall sub_37ADB10(__int64 *a1, __int64 a2, __m128i a3)
{
  int v3; // eax
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int16 v11; // [rsp+0h] [rbp-60h] BYREF
  __int64 v12; // [rsp+8h] [rbp-58h]
  __int16 v13; // [rsp+10h] [rbp-50h] BYREF
  __int64 v14; // [rsp+18h] [rbp-48h]

  v3 = *(unsigned __int16 *)(a2 + 96);
  v4 = *(_QWORD *)(a2 + 104);
  v11 = v3;
  v12 = v4;
  if ( (_WORD)v3 )
  {
    if ( (unsigned __int16)(v3 - 17) > 0xD3u )
    {
      v13 = v3;
      v14 = v4;
LABEL_4:
      if ( (_WORD)v3 == 1 || (unsigned __int16)(v3 - 504) <= 7u )
        BUG();
      v5 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v3 - 16];
      if ( v5 && (v5 & 7) == 0 )
        return sub_37AD520(a1, a2, a3);
      return sub_3461110(a3, *a1, a2, a1[1]);
    }
    LOWORD(v3) = word_4456580[v3 - 1];
    v7 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v11) )
    {
      v14 = v4;
      v13 = 0;
      goto LABEL_11;
    }
    LOWORD(v3) = sub_3009970((__int64)&v11, a2, v8, v9, v10);
  }
  v13 = v3;
  v14 = v7;
  if ( (_WORD)v3 )
    goto LABEL_4;
LABEL_11:
  if ( sub_3007260((__int64)&v13) && (sub_3007260((__int64)&v13) & 7) == 0 )
    return sub_37AD520(a1, a2, a3);
  return sub_3461110(a3, *a1, a2, a1[1]);
}
