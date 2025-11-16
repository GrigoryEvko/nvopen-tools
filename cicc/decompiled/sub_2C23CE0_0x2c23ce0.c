// Function: sub_2C23CE0
// Address: 0x2c23ce0
//
__int64 __fastcall sub_2C23CE0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  bool v3; // zf
  int v4; // eax
  __int16 v5; // dx
  __int64 v6; // rax
  char v7; // r14
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  int i; // r14d
  __int64 v13; // rax
  __int64 result; // rax
  __int64 v15; // rdx
  int v16; // [rsp+Ch] [rbp-64h]
  __int64 v17; // [rsp+18h] [rbp-58h] BYREF
  __int64 v18; // [rsp+20h] [rbp-50h] BYREF
  int v19; // [rsp+28h] [rbp-48h]
  __int64 v20; // [rsp+30h] [rbp-40h]
  __int16 v21; // [rsp+38h] [rbp-38h]
  char v22; // [rsp+3Ah] [rbp-36h]

  v2 = *(_QWORD *)(a2 + 904);
  v3 = *(_BYTE *)(a1 + 152) == 5;
  v4 = *(_DWORD *)(v2 + 104);
  v5 = *(_WORD *)(v2 + 108);
  v18 = v2;
  v19 = v4;
  v6 = *(_QWORD *)(v2 + 96);
  v21 = v5;
  v20 = v6;
  v22 = *(_BYTE *)(v2 + 110);
  if ( v3 )
    *(_DWORD *)(v2 + 104) = sub_2C1A110(a1);
  v17 = *(_QWORD *)(a1 + 88);
  if ( v17 )
    sub_2AAAFA0(&v17);
  sub_2BF1A90(a2, (__int64)&v17);
  sub_9C6650(&v17);
  v7 = sub_2C1A9C0(a1);
  if ( v7 && !(unsigned __int8)sub_2C46C30(a1 + 96) && !sub_2C1A990(a1) )
    v7 = sub_2C1A9B0(a1);
  if ( (unsigned __int8)sub_2C1A8C0(a1) )
  {
    v16 = *(_DWORD *)(a2 + 8);
    if ( v16 )
    {
      for ( i = 0; i != v16; ++i )
      {
        LODWORD(v17) = i;
        BYTE4(v17) = 0;
        v13 = sub_2C1A8F0(a1, a2, (unsigned int *)&v17);
        LODWORD(v17) = i;
        BYTE4(v17) = 0;
        sub_2AC6E90(a2, a1 + 96, v13, (unsigned int *)&v17);
      }
    }
LABEL_10:
    result = v18;
    *(_DWORD *)(v18 + 104) = v19;
    *(_QWORD *)(result + 96) = v20;
    *(_WORD *)(result + 108) = v21;
    *(_BYTE *)(result + 110) = v22;
  }
  else
  {
    v15 = sub_2C22B80(a1, a2, v8, v9, v10, v11);
    switch ( *(_BYTE *)(a1 + 160) )
    {
      case 1:
      case 2:
      case 3:
      case 4:
      case 6:
      case 7:
      case 9:
      case 0x21:
      case 0x23:
      case 0x25:
      case 0x4E:
      case 0x4F:
        goto LABEL_10;
      default:
        sub_2BF26E0(a2, a1 + 96, v15, v7);
        result = sub_2C135A0(&v18);
        break;
    }
  }
  return result;
}
