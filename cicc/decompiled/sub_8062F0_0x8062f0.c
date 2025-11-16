// Function: sub_8062F0
// Address: 0x8062f0
//
__int64 __fastcall sub_8062F0(__int64 *a1)
{
  __int64 v1; // r15
  __int64 v2; // rbx
  int *v3; // rax
  int v4; // edx
  _BYTE *v5; // rax
  char v6; // al
  int *v7; // rax
  int v8; // edx
  __int64 result; // rax
  _BYTE v10[80]; // [rsp+0h] [rbp-50h] BYREF

  v1 = *(_QWORD *)(qword_4F04C50 + 48LL);
  v2 = *(_QWORD *)(qword_4F04C50 + 40LL);
  v3 = *(int **)(*(_QWORD *)(qword_4F04C50 + 80LL) + 80LL);
  v4 = *v3;
  LOWORD(v3) = *((_WORD *)v3 + 2);
  dword_4F07508[0] = v4;
  LOWORD(dword_4F07508[1]) = (_WORD)v3;
  *(_QWORD *)dword_4D03F38 = *(_QWORD *)dword_4F07508;
  v5 = sub_726B30(11);
  *a1 = (__int64)v5;
  sub_7E1740((__int64)v5, (__int64)v10);
  if ( v1 )
  {
    while ( 1 )
    {
      v6 = *(_BYTE *)(v1 + 8);
      if ( v6 != 2 )
        break;
      sub_7FCE40(v1, v2, 1u, 0, 0, (__int64)v10);
      v1 = *(_QWORD *)v1;
      if ( !v1 )
        goto LABEL_9;
    }
    while ( v6 == 1 )
    {
      sub_7FCE40(v1, v2, 0, 0, a1[2], (__int64)v10);
      v1 = *(_QWORD *)v1;
      if ( !v1 )
        break;
      v6 = *(_BYTE *)(v1 + 8);
    }
  }
LABEL_9:
  v7 = *(int **)(qword_4F04C50 + 80LL);
  v8 = *v7;
  LOWORD(v7) = *((_WORD *)v7 + 2);
  dword_4D03F38[0] = v8;
  LOWORD(dword_4D03F38[1]) = (_WORD)v7;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)dword_4D03F38;
  result = *a1;
  if ( !*(_QWORD *)(*a1 + 72) )
    *a1 = 0;
  return result;
}
