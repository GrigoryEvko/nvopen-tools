// Function: sub_130D990
// Address: 0x130d990
//
__int64 __fastcall sub_130D990(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, char a5, char a6)
{
  unsigned __int64 v11; // rsi
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rdi
  _BYTE *v14; // rsi
  _BYTE *v15; // rbx
  _BYTE *v16; // rdi
  __int64 v17; // rax
  unsigned __int64 v20; // [rsp+8h] [rbp-38h]

  sub_1341E90(a1, a4);
  v11 = a3[2] & 0xFFFFFFFFFFFFF000LL;
  if ( a6 && a5 )
  {
    v12 = v11 + 0x2000;
    v13 = a3[1] & 0xFFFFFFFFFFFFF000LL;
    goto LABEL_4;
  }
  v12 = v11 + 4096;
  v13 = a3[1] & 0xFFFFFFFFFFFFF000LL;
  if ( a6 )
  {
LABEL_4:
    v14 = (_BYTE *)(v13 + v11);
    if ( !a5 )
      goto LABEL_5;
LABEL_9:
    v16 = (_BYTE *)(v13 - 4096);
    v15 = v16;
    if ( *(__int64 (__fastcall ***)(int, int, int, int, int, int, int))(a2 + 8) != &off_49E8020 )
      goto LABEL_6;
LABEL_10:
    v20 = v12;
    sub_1341250(v16, v14);
    v12 = v20;
    goto LABEL_6;
  }
  v14 = 0;
  if ( a5 )
    goto LABEL_9;
LABEL_5:
  v15 = (_BYTE *)v13;
  v16 = 0;
  if ( *(__int64 (__fastcall ***)(int, int, int, int, int, int, int))(a2 + 8) == &off_49E8020 )
    goto LABEL_10;
LABEL_6:
  v17 = a3[2];
  a3[1] = v15;
  *a3 &= ~0x10000uLL;
  a3[2] = v12 | v17 & 0xFFF;
  return sub_1341BA0(a1, a4, a3, 232, 0);
}
