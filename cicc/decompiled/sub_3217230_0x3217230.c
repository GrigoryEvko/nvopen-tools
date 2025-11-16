// Function: sub_3217230
// Address: 0x3217230
//
char __fastcall sub_3217230(__int64 a1, __int64 *a2)
{
  __int64 v2; // r12
  int v3; // ebx
  __int64 v4; // r15
  char v5; // r8
  bool v6; // al
  __int64 v7; // rsi
  int v8; // ecx
  char result; // al
  unsigned __int8 *v10; // rax
  unsigned __int8 *v11; // r9
  int v12; // eax
  int v13; // eax
  char v14; // al
  unsigned __int8 *v15; // [rsp+0h] [rbp-50h]
  int v16; // [rsp+Ch] [rbp-44h]
  __int64 v17; // [rsp+10h] [rbp-40h]
  __int64 v18; // [rsp+18h] [rbp-38h]

  *(_BYTE *)(a1 + 26) = 0;
  *(_BYTE *)(a1 + 24) = 0;
  v2 = *a2;
  v17 = a2[55];
  v18 = a2[54];
  v3 = sub_31DB780(*(_QWORD *)(a1 + 8), a2);
  v4 = sub_31DA6B0(*(_QWORD *)(a1 + 8));
  v16 = *(_DWORD *)(v4 + 940);
  if ( (*(_BYTE *)(v2 + 2) & 8) == 0 )
  {
    *(_BYTE *)(a1 + 25) = 0;
    goto LABEL_3;
  }
  v10 = (unsigned __int8 *)sub_B2E500(v2);
  v11 = sub_BD3990(v10, (__int64)a2);
  if ( *v11 >= 4u )
    v11 = 0;
  if ( (*(_BYTE *)(v2 + 2) & 8) != 0
    && (v15 = v11, v12 = sub_B2A630((__int64)v11), v11 = v15, !v12)
    && ((v13 = sub_A746B0((_QWORD *)(v2 + 120)), v11 = v15, v13)
     || (v14 = sub_B2D610(v2, 41), v11 = v15, !v14)
     || (*(_BYTE *)(v2 + 2) & 8) != 0) )
  {
    *(_BYTE *)(a1 + 25) = 1;
  }
  else
  {
    *(_BYTE *)(a1 + 25) = 0;
    if ( v16 == 255 || v18 == v17 )
      goto LABEL_3;
  }
  if ( !v11 )
  {
LABEL_3:
    *(_BYTE *)(a1 + 24) = 0;
    v5 = 0;
    v6 = 0;
    goto LABEL_4;
  }
  *(_BYTE *)(a1 + 24) = 1;
  v5 = 1;
  v6 = *(_DWORD *)(v4 + 944) != 255;
LABEL_4:
  *(_BYTE *)(a1 + 26) = v6;
  v7 = *(_QWORD *)(a2[3] + 152);
  v8 = *(_DWORD *)(v7 + 336);
  if ( v8 )
  {
    result = v8 == 7 || (v8 & 0xFFFFFFFD) == 1;
    if ( result || v8 == 4 && (result = *(_DWORD *)(v7 + 344) != 6 && *(_DWORD *)(v7 + 344) != 0) != 0 )
      result = v5 | (v3 != 0);
  }
  else
  {
    result = sub_31DB810(*(_QWORD *)(a1 + 8)) & (v3 != 0);
  }
  *(_BYTE *)(a1 + 27) = result;
  return result;
}
