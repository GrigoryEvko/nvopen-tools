// Function: sub_11FAD60
// Address: 0x11fad60
//
__int64 __fastcall sub_11FAD60(int a1, char a2, __int64 a3, _DWORD *a4)
{
  __int64 result; // rax
  _QWORD *v6; // rdi
  int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // rdi
  _QWORD *v10; // rdi
  int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // rdi
  __int64 v14; // [rsp-18h] [rbp-18h]
  __int64 v15; // [rsp-10h] [rbp-10h]

  switch ( a1 )
  {
    case 0:
      v10 = *(_QWORD **)a3;
      v11 = *(unsigned __int8 *)(a3 + 8);
      if ( (unsigned int)(v11 - 17) > 1 )
      {
        v13 = sub_BCB2A0(v10);
      }
      else
      {
        BYTE4(v14) = (_BYTE)v11 == 18;
        LODWORD(v14) = *(_DWORD *)(a3 + 32);
        v12 = (__int64 *)sub_BCB2A0(v10);
        v13 = sub_BCE1B0(v12, v14);
      }
      result = sub_AD64C0(v13, 0, 0);
      break;
    case 1:
      *a4 = a2 == 0 ? 34 : 38;
      result = 0;
      break;
    case 2:
      *a4 = 32;
      result = 0;
      break;
    case 3:
      *a4 = a2 == 0 ? 35 : 39;
      result = 0;
      break;
    case 4:
      *a4 = a2 == 0 ? 36 : 40;
      result = 0;
      break;
    case 5:
      *a4 = 33;
      result = 0;
      break;
    case 6:
      *a4 = a2 == 0 ? 37 : 41;
      result = 0;
      break;
    case 7:
      v6 = *(_QWORD **)a3;
      v7 = *(unsigned __int8 *)(a3 + 8);
      if ( (unsigned int)(v7 - 17) > 1 )
      {
        v9 = sub_BCB2A0(v6);
      }
      else
      {
        BYTE4(v15) = (_BYTE)v7 == 18;
        LODWORD(v15) = *(_DWORD *)(a3 + 32);
        v8 = (__int64 *)sub_BCB2A0(v6);
        v9 = sub_BCE1B0(v8, v15);
      }
      result = sub_AD64C0(v9, 1, 0);
      break;
    default:
      BUG();
  }
  return result;
}
