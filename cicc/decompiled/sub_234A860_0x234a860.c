// Function: sub_234A860
// Address: 0x234a860
//
void __fastcall sub_234A860(__int64 a1, __int64 *a2)
{
  unsigned __int64 v2; // rbp
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  unsigned __int64 v6; // r8
  __int64 v7; // rax
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // rdx
  __int64 v10; // rax
  unsigned __int64 v11[5]; // [rsp-28h] [rbp-28h] BYREF

  v3 = *a2;
  if ( *(_BYTE *)(a1 + 24) )
  {
    v11[4] = v2;
    v6 = *(_QWORD *)a1;
    *(_QWORD *)a1 = v3;
    v7 = a2[1];
    v8 = *(_QWORD *)(a1 + 8);
    v9 = *(_QWORD *)(a1 + 16);
    v11[0] = v6;
    *(_QWORD *)(a1 + 8) = v7;
    v10 = a2[2];
    v11[1] = v8;
    *(_QWORD *)(a1 + 16) = v10;
    *a2 = 0;
    a2[1] = 0;
    a2[2] = 0;
    v11[2] = v9;
    sub_234A6B0(v11);
  }
  else
  {
    *(_QWORD *)a1 = v3;
    v4 = a2[1];
    *a2 = 0;
    *(_QWORD *)(a1 + 8) = v4;
    v5 = a2[2];
    a2[1] = 0;
    a2[2] = 0;
    *(_QWORD *)(a1 + 16) = v5;
    *(_BYTE *)(a1 + 24) = 1;
  }
}
