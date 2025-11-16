// Function: sub_2FE5260
// Address: 0x2fe5260
//
__int64 __fastcall sub_2FE5260(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 (__fastcall *v4)(__int64, __int64, unsigned int); // rax
  int v5; // eax
  int v6; // eax
  __int64 v7; // rax
  char v8; // cl
  __int64 v9; // rax
  __int64 v11; // [rsp+0h] [rbp-10h] BYREF
  char v12; // [rsp+8h] [rbp-8h]

  v4 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)a1 + 32LL);
  if ( v4 == sub_2D42F30 )
  {
    v5 = sub_AE2980(a2, 0)[1];
    switch ( v5 )
    {
      case 1:
        v6 = 2;
        break;
      case 2:
        v6 = 3;
        break;
      case 4:
        v6 = 4;
        break;
      case 8:
        v6 = 5;
        break;
      case 16:
        v6 = 6;
        break;
      case 32:
        v6 = 7;
        break;
      case 64:
        v6 = 8;
        break;
      case 128:
        v6 = 9;
        break;
      default:
        goto LABEL_10;
    }
  }
  else
  {
    v6 = ((unsigned __int16 (__fastcall *)(__int64, __int64, _QWORD, __int64, __int64))v4)(a1, a2, 0, a4, a2);
    if ( (unsigned __int16)v6 <= 1u || (unsigned __int16)(v6 - 504) <= 7u )
LABEL_10:
      BUG();
  }
  v7 = 16LL * (v6 - 1);
  v8 = byte_444C4A0[v7 + 8];
  v9 = *(_QWORD *)&byte_444C4A0[v7];
  v12 = v8;
  v11 = v9;
  return sub_CA1930(&v11);
}
