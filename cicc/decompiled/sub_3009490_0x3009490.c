// Function: sub_3009490
// Address: 0x3009490
//
__int64 __fastcall sub_3009490(unsigned __int16 *a1, unsigned int a2, __int64 a3)
{
  unsigned int v3; // r14d
  int v6; // edx
  __int64 *v7; // r13
  __int16 v8; // di
  int v9; // esi
  int v10; // eax
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v14; // rax
  __int64 v15; // [rsp+0h] [rbp-30h]

  v6 = *a1;
  v7 = (__int64 *)**((_QWORD **)a1 + 1);
  if ( (_WORD)v6 )
  {
    v8 = a2;
    v9 = word_4456340[v6 - 1];
    BYTE4(v15) = (unsigned __int16)(v6 - 176) <= 0x34u;
    LODWORD(v15) = v9;
    if ( (unsigned __int16)(v6 - 176) <= 0x34u )
      goto LABEL_3;
LABEL_6:
    LOWORD(v10) = sub_2D43050(v8, v9);
    if ( (_WORD)v10 )
      goto LABEL_4;
LABEL_7:
    v10 = sub_3009450(v7, a2, a3, v15, v11, v12);
    HIWORD(v3) = HIWORD(v10);
    goto LABEL_4;
  }
  v14 = sub_3007240((__int64)a1);
  v8 = a2;
  v9 = v14;
  LODWORD(v15) = v14;
  BYTE4(v15) = BYTE4(v14);
  if ( !BYTE4(v14) )
    goto LABEL_6;
LABEL_3:
  LOWORD(v10) = sub_2D43AD0(v8, v9);
  if ( !(_WORD)v10 )
    goto LABEL_7;
LABEL_4:
  LOWORD(v3) = v10;
  return v3;
}
