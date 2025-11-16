// Function: malloc_usable_size
// Address: 0x1308a20
//
__int64 __fastcall malloc_usable_size(unsigned __int64 a1)
{
  unsigned __int64 v2; // rdi
  _BYTE *v3; // rbx
  unsigned __int64 v4; // rcx
  unsigned __int64 *v5; // rax
  unsigned __int64 v6; // rsi
  _QWORD *v7; // r12
  unsigned __int64 v9; // rdx
  __int64 v10; // rax
  _QWORD *v11; // rdx
  unsigned int i; // r9d
  _BYTE *v13; // r8
  _BYTE v14[400]; // [rsp+0h] [rbp-190h] BYREF

  if ( !unk_4F96B58 )
  {
    if ( !a1 )
      return 0;
LABEL_10:
    v3 = v14;
    sub_130D500(v14);
    v2 = 0;
    goto LABEL_6;
  }
  if ( __readfsbyte(0xFFFFF8C8) )
  {
    v10 = sub_1313D30(__readfsqword(0) - 2664, 0);
    v2 = v10;
    if ( !a1 )
      return 0;
    if ( v10 )
      goto LABEL_5;
    goto LABEL_10;
  }
  if ( !a1 )
    return 0;
  v2 = __readfsqword(0) - 2664;
LABEL_5:
  v3 = (_BYTE *)(v2 + 432);
LABEL_6:
  v4 = a1 & 0xFFFFFFFFC0000000LL;
  v5 = (unsigned __int64 *)&v3[(a1 >> 26) & 0xF0];
  v6 = *v5;
  if ( (a1 & 0xFFFFFFFFC0000000LL) == *v5 )
  {
    v7 = (_QWORD *)(v5[1] + ((a1 >> 9) & 0x1FFFF8));
  }
  else if ( v4 == *((_QWORD *)v3 + 32) )
  {
    v9 = *((_QWORD *)v3 + 33);
LABEL_14:
    *((_QWORD *)v3 + 32) = v6;
    *((_QWORD *)v3 + 33) = v5[1];
    v7 = (_QWORD *)(v9 + ((a1 >> 9) & 0x1FFFF8));
    *v5 = v4;
    v5[1] = v9;
  }
  else
  {
    v11 = v3 + 272;
    for ( i = 1; i != 8; ++i )
    {
      if ( v4 == *v11 )
      {
        v13 = &v3[16 * i];
        v3 += 16 * i - 16;
        v9 = *((_QWORD *)v13 + 33);
        *((_QWORD *)v13 + 32) = *((_QWORD *)v3 + 32);
        *((_QWORD *)v13 + 33) = *((_QWORD *)v3 + 33);
        goto LABEL_14;
      }
      v11 += 2;
    }
    v7 = (_QWORD *)sub_130D370(v2, &unk_5060AE0, v3, a1, 1, 0);
  }
  return qword_505FA40[HIWORD(*v7)];
}
