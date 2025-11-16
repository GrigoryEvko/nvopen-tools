// Function: sub_29885D0
// Address: 0x29885d0
//
__int64 __fastcall sub_29885D0(__int64 a1, __int64 a2, __int64 a3, int a4, unsigned __int8 a5)
{
  unsigned int v5; // r14d
  unsigned int v6; // r15d
  __int64 v7; // r13
  char v8; // dl
  int v9; // eax
  char v12; // al
  __int64 v13; // rax
  unsigned int v14; // eax
  char v16; // [rsp+Ch] [rbp-54h]
  __int64 v17; // [rsp+10h] [rbp-50h] BYREF
  _DWORD v18[5]; // [rsp+18h] [rbp-48h] BYREF
  int v19; // [rsp+2Ch] [rbp-34h]

  v7 = *(_QWORD *)(a2 + 16);
  if ( !a5 )
    v7 = *(_QWORD *)(a2 + 8);
  HIBYTE(v19) = 0;
  *(_WORD *)((char *)&v19 + 1) = 0;
  if ( (*(_DWORD *)(a3 + 4) & 0x7FFFFFF) == 3 )
  {
    v7 = *(_QWORD *)(a3 - 96);
    v12 = sub_BC8C50(a3, &v17, v18);
    v8 = 0;
    if ( v12 )
    {
      v5 = v18[0];
      v6 = v17;
      v8 = 1;
      if ( a5 == a4 )
        goto LABEL_5;
    }
    else if ( a5 == a4 )
    {
      goto LABEL_5;
    }
    v16 = v8;
    v13 = sub_F5E1A0(v7);
    v8 = v16;
    v7 = v13;
    if ( v16 )
    {
      v14 = v6;
      v6 = v5;
      v5 = v14;
    }
  }
  else
  {
    v8 = 0;
    v5 = 0;
    v6 = 0;
  }
LABEL_5:
  *(_QWORD *)&v18[3] = __PAIR64__(v5, v6);
  LOBYTE(v19) = v8;
  *(_QWORD *)(a1 + 8) = __PAIR64__(v5, v6);
  v9 = v19;
  *(_QWORD *)a1 = v7;
  *(_DWORD *)(a1 + 16) = v9;
  return a1;
}
