// Function: sub_685920
// Address: 0x685920
//
__int64 __fastcall sub_685920(_DWORD *a1, FILE *a2, unsigned __int8 a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  FILE *v6; // r14
  __int64 flags; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  unsigned __int8 v10; // dl
  _DWORD *v11; // r13
  _DWORD *v13; // r13
  _BYTE v14[33]; // [rsp+Fh] [rbp-21h] BYREF

  v4 = sub_87D1A0(a2, v14);
  if ( v14[0] > 0xCu )
  {
    if ( v14[0] == 59 )
    {
      v5 = *(_QWORD *)(*(_QWORD *)(v4 + 200) + 208LL);
      v6 = (FILE *)(v5 + 64);
      if ( v5 )
      {
        flags = (unsigned int)v6->_flags;
        v8 = (unsigned int)dword_4F077C8;
        if ( (_DWORD)flags != dword_4F077C8 )
          goto LABEL_5;
LABEL_12:
        v10 = a3;
        if ( *((unsigned __int16 *)&v6->_flags + 2) != (unsigned __int64)unk_4F077CC )
          goto LABEL_6;
        goto LABEL_8;
      }
    }
LABEL_7:
    v10 = a3;
    goto LABEL_8;
  }
  if ( v14[0] <= 0xAu && (unsigned __int8)(v14[0] - 6) > 1u )
    goto LABEL_7;
  v6 = (FILE *)(sub_72A270(v4, v14[0]) + 64);
  flags = (unsigned int)v6->_flags;
  v8 = (unsigned int)dword_4F077C8;
  if ( (_DWORD)flags == dword_4F077C8 )
    goto LABEL_12;
LABEL_5:
  v9 = flags - v8;
  v10 = a3;
  if ( v9 )
  {
LABEL_6:
    v11 = sub_67D610(0xD29u, a1, v10);
    sub_67F060((__int64)v11, (__int64)a2);
    sub_67EF30((__int64)v11, v6);
    return sub_685910((__int64)v11, v6);
  }
LABEL_8:
  v13 = sub_67D610(0xF7u, a1, v10);
  sub_67F060((__int64)v13, (__int64)a2);
  return sub_685910((__int64)v13, a2);
}
