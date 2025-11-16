// Function: sub_130D840
// Address: 0x130d840
//
__int64 __fastcall sub_130D840(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, char a5, char a6, char a7)
{
  __int64 v8; // rax
  char v12; // cl
  unsigned __int64 v13; // rbx
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rbx
  void *v16; // r8
  __int64 (__fastcall **v17)(int, int, int, int, int, int, int); // rax
  void *v18; // rsi
  __int64 v19; // rax
  __int64 result; // rax
  char v21; // [rsp+Ch] [rbp-44h]
  __int64 v22; // [rsp+10h] [rbp-40h]
  __int64 v23; // [rsp+18h] [rbp-38h]
  unsigned __int64 v24; // [rsp+18h] [rbp-38h]

  v8 = a2;
  v12 = a7;
  if ( a7 )
  {
    v23 = a4;
    sub_1341E90(a1, a4);
    v12 = a7;
    v8 = a2;
    a4 = v23;
  }
  v13 = a3[2] & 0xFFFFFFFFFFFFF000LL;
  v14 = a3[1] & 0xFFFFFFFFFFFFF000LL;
  if ( a5 && a6 )
  {
    v15 = v13 - 0x2000;
LABEL_6:
    v16 = (void *)(a3[1] & 0xFFFFFFFFFFFFF000LL);
    v14 += 4096LL;
    goto LABEL_7;
  }
  v15 = v13 - 4096;
  if ( a5 )
    goto LABEL_6;
  v16 = 0;
LABEL_7:
  v17 = *(__int64 (__fastcall ***)(int, int, int, int, int, int, int))(v8 + 8);
  v18 = (void *)(v14 + v15);
  if ( !a6 )
    v18 = 0;
  if ( v17 == &off_49E8020 )
  {
    v21 = v12;
    v22 = a4;
    v24 = v14;
    sub_1341240(v16, v18);
    v12 = v21;
    a4 = v22;
    v14 = v24;
  }
  v19 = a3[2];
  *a3 |= 0x10000uLL;
  a3[1] = v14;
  result = v19 & 0xFFF;
  a3[2] = result | v15;
  if ( v12 )
    return sub_1341BA0(a1, a4, a3, 232, 0);
  return result;
}
