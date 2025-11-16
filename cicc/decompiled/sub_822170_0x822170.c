// Function: sub_822170
// Address: 0x822170
//
__int64 __fastcall sub_822170(__int64 a1)
{
  int v1; // eax
  char v2; // dl
  int v3; // eax
  __int64 result; // rax
  __int64 v5; // rbx
  __int64 v6; // rbx
  char v7; // [rsp+7h] [rbp-29h] BYREF
  __int64 v8; // [rsp+8h] [rbp-28h]
  __int16 v9; // [rsp+10h] [rbp-20h]
  char src; // [rsp+12h] [rbp-1Eh] BYREF
  int v11; // [rsp+13h] [rbp-1Dh]
  __int16 v12; // [rsp+17h] [rbp-19h]
  char v13; // [rsp+19h] [rbp-17h]
  int v14; // [rsp+1Ah] [rbp-16h]
  char v15; // [rsp+1Eh] [rbp-12h]
  char v16; // [rsp+1Fh] [rbp-11h]

  v1 = *(_DWORD *)(a1 + 4);
  v2 = *(_BYTE *)(a1 + 10);
  v15 = 34;
  src = 34;
  v11 = v1;
  LOWORD(v1) = *(_WORD *)(a1 + 8);
  v13 = v2;
  v12 = v1;
  if ( (_BYTE)v1 == 48 )
    LOBYTE(v12) = 32;
  v3 = *(_DWORD *)(a1 + 20);
  v16 = 0;
  v7 = 34;
  v14 = v3;
  v8 = *(_QWORD *)(a1 + 11);
  v9 = 34;
  if ( dword_4D03C90 )
  {
    v5 = *(_QWORD *)(qword_4F194A8 + 88);
    *(_QWORD *)(v5 + 16) = sub_819EB0(&src, 0);
    v6 = *(_QWORD *)(qword_4F194A0 + 88);
    result = (__int64)sub_819EB0(&v7, 0);
    *(_QWORD *)(v6 + 16) = result;
  }
  else
  {
    qword_4F194A8 = sub_822070(&src, "__DATE__", 1, 1);
    result = sub_822070(&v7, "__TIME__", 1, 1);
    qword_4F194A0 = result;
  }
  return result;
}
